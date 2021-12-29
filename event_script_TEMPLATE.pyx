import db
import event
from datetime import datetime, timedelta
import math
import sys
import statistics
import numpy as np
import pandas as pd
import CONSTANTS
import cython
import time
np.seterr(divide='ignore', invalid='ignore')


### HOW IT WORKS
# function takes an 'input_settings' object and returns an 'event_table' object
# all parts where something needs to be changed are marked with #***CUSTOMISE
# the function loops through data (data is stored in 2 vectors - 'dt' and 'px') using a while loop - start of loop marked as '########MAIN EVENT CALCULATION LOOP########'
# signal is calculated through combining several criteria variables (cr0, cr1, ...) - marked as #SIGNAL_TRIGGER
# direction of the signal (long = 1, short = -1) is stored in a variable event_direction for each single event / loop iteration
# when a signal is triggered it is added to vectors: 'event_dates', 'event_signals', 'event_directions' - in section marked as #ADDING AN EVENT / SIGNAL
# these vectors are stored in the output object - 'event_table'
# there are 2 objects which store the calculation trail for signals: 'other_clue_parameters_trail' and 'calculation_trail'
###


#SETTINGS
XXX_input_setts = {
        ticker_key:                     "QQQ",
        event_id_key:                   "XXX - RENAME", #CUSTOMISE
        "from_date":                    '31-12-2019',
        "to_date":                      0,

        "move_size_min":                0.0,			#CUSTOMISE
        "move_size_max_ratio_of_min":   0.0, 			#CUSTOMISE

        "extremum_time_days_min":       90,				#CUSTOMISE
        "extremum_time_days_max":       10000, 			#CUSTOMISE

        "no_overlap_time":              1*1440,
        "trade_same_dir_as_extr_or_no": -1,
        "mins_maxes_or_both":           -1,

        'option_type':                  'put',
        'option_strike_pct':            1.00,
        'option_tenor':                 15,
        'option_long_or_short':         -1,
        'option_IV_if_not_avail_in_db': 0.30,
        'event2_within':                1440*30,
}




#EVENT/SIGNAL CALCULATION FUNCTION
cpdef event_XXX(input_settings): #***CUSTOMISE: CHANGE FUNCTION NAME
    cdef start_time = time.time()

    cdef int print_or_no = CONSTANTS.print_or_no
    if print_or_no == 1: print("\n---- STARTING XXX CALC..INPUT SETTS: ", input_settings, "")

    cdef list event_dates = []
    cdef list event_prices_loc = []
    cdef list event_directions = []
    cdef list event_tickers = []


	#***********CUSTOMISE - START - CALCULATION TRAIL CRITERIA***********
    cdef list event_start_dates   = [] #do not change
    cdef list event_move_sizes_from_previous_extremum    = []   #from min between curr max
    cdef list number_of_days_extremum_over = []
    cdef list other_clue_params_slope = []
    #***********CUSTOMISE - END - CALCULATION TRAIL***********

    cdef list tickers   = []

    if input_settings[event.ticker_key] not in db.data_db.keys():
        tickers = [input_settings[event.ticker_key]]
        db.fetch_necessary_data_from_db(tickers)

    cdef data = db.data_db[input_settings[event.ticker_key]]
    cdef dts = data[db.in_memory_db_date_key]

    if len(dts) == 0: return
    if type(data[db.in_memory_db_date_key][0]) != datetime: dts = dts.dt.to_pydatetime()

    cdef dt = dts
    cdef px = data[db.price_key].to_numpy()
    if len(px) == 0: return

    ###############
    if print_or_no == 1:
        cnt = 0
        print("TYPES BEFORE COMPRESSION")
        print("DT: ", type(data[db.in_memory_db_date_key]))
        print("PX: ", type(data[db.price_key]))

        for el,el2 in zip(dt,px):
            print(el,el2,type(el),type(el2))
            cnt += 1
            if cnt > 10: break

    #COMPRESS DATA
    if len(dt) > CONSTANTS.COMPRESSION_THRESHOLD and db.data_db[input_settings[event.ticker_key]][CONSTANTS.compressed_flag_key].tolist()[0] == 0:
        dt,px = event.event_table.compress_data_fully(event.event_table,db.data_db[input_settings[event.ticker_key]],5)
        #print("LEN AFTER: ",len(dt))
        #NEED TO CHANGE THEM BACK TO ORIGINAL DATATYPES
        db.data_db[input_settings[event.ticker_key]] = pd.DataFrame({
            'Date': pd.Series(dt.tolist()), #np.asarray(dt.tolist(), dtype=datetime.datetime),
            'Close': pd.Series(list(px)), #.tolist(), np.asarray(list(px), dtype=np.float64),
            CONSTANTS.compressed_flag_key: pd.Series([1]*len(dt)),
        })
        print("overwrote the data in db")
    else:
        print("     in event calc func - data is under 50k points - no need to compress")

    if print_or_no == 1:
        cnt = 0
        for el,el2 in zip(db.data_db[input_settings[event.ticker_key]]['Date'],db.data_db[input_settings[event.ticker_key]]['Close']):
            print(el,el2,type(el),type(el2))
            cnt += 1
            if cnt > 10: break
    ###############

    # unpack settings to a more convenient format - the following 5 variable are always part of settings
    input_start_date = input_settings['from_date']
    input_end_date = input_settings['to_date']
    cdef    int    input_no_overlap_time = input_settings['no_overlap_time']
    cdef    int    input_trade_same_dir_as_extr_or_no = input_settings['trade_same_dir_as_extr_or_no']
    cdef    int    input_mins_maxes_or_both = input_settings['mins_maxes_or_both']

    #***********CUSTOMISE - INPUT PARAMETERS***********
    cdef    int     input_min_size_of_extremum = int(input_settings['extremum_time_days_min'])
    cdef    int     input_max_size_of_extremum = int(input_settings['extremum_time_days_max'])

	#ADD OWN INPUT PARAMETERS HERE

	#***********CUSTOMISE END - INPUT PARAMETERS*******



    cdef    long    i       = 0         #counter for main loop
    cdef    long    lendt   = len(dt)

    if type(input_start_date) is int: input_start_date = datetime.min
    if type(input_start_date) is str: input_start_date = datetime.strptime(input_start_date, '%d-%m-%Y')

    if type(input_end_date) is int: input_end_date = datetime.min
    if type(input_end_date) is str: input_end_date = datetime.strptime(input_end_date, '%d-%m-%Y')

    if input_start_date > datetime.min: i = np.searchsorted(dt, input_start_date)
    if input_end_date > datetime.min: lendt = np.searchsorted(dt, input_end_date)

    cdef int trade_on_business_days_only = 1
    cdef str asset_class = db.other_attr_db[db.db_asset_class_record_key]
    if asset_class == "cryptocurrency": trade_on_business_days_only = 0


	#***********CUSTOMISE***********
    #CRITERIA - ADD MORE IF NEEDED
    cdef    int    cr0 = 0
    cdef    int    cr1 = 0
    cdef    int    cr2 = 0
    cdef    int    cr3 = 0
	#***********CUSTOMISE END*******

    cdef    int    event_direction = 0 #do not change

	#***********CUSTOMISE***********
    cdef    long        lookback_start_idx = 0
    cdef    double      move_vertical_sz = 0.0
    cdef    double      prev_local_min = 0.0

    cdef   prev_higher_px_date = datetime.min
    cdef   time_of_max_idx = datetime.min

    cdef double prev_local_max = 0.0
	#***********CUSTOMISE END*******


    ########MAIN EVENT CALCULATION LOOP########
    while i < lendt:
        if i < 2:
            i+=1
            continue

        if print_or_no == 1:
            if i % 10000 == 0 or (i >=2 and i < 3): print("in event calc: ", i)


        #reset signal criteria on each loop
        cr0 = 0 #SIGNAL CALCULATION CRITERIA 0
        cr1 = 0 #SIGNAL CALCULATION CRITERIA 1
        cr2 = 1 #SIGNAL CALCULATION CRITERIA 2
        cr3 = 0 #SIGNAL CALCULATION CRITERIA  - FOR TIMEOUT BETWEEN SIGNALS
		#cr4 = 0, cr5 = 0,...
		#***********CUSTOMISE: ADD MORE CRITERIA HERE IF NEEDED*******

        if len(event_dates) == 0:
            cr3 = 1
        else:
            if ((dt[i] - max(event_dates)) >= timedelta(minutes=input_no_overlap_time)): cr3 = 1

        if cr3 == 0:
            i += 1
            continue

		#***********CUSTOMISE: CALCULATION OF SIGNAL GOES HERE - START*******

        #LONG SIGNALS
        if input_mins_maxes_or_both == 1:
			#CALCULATE THE SIGNAL HERE
            idx_of_previously_higher_px_arr = np.where(px[:i] >= px[i])[0]
            if len(idx_of_previously_higher_px_arr) == 0:
                idx_of_previously_higher_px = 0
                prev_higher_px = float('NaN')
                prev_higher_px_date = datetime.min
            else:
                idx_of_previously_higher_px = int(idx_of_previously_higher_px_arr[len(idx_of_previously_higher_px_arr)-1])
                prev_higher_px = px[idx_of_previously_higher_px]
                prev_higher_px_date = dt[idx_of_previously_higher_px]


            if prev_higher_px_date == datetime.min and input_max_size_of_extremum == 10000:
                cr0 = 1
            else:
                cr0 = int(1 * (dt[i] - prev_higher_px_date <= timedelta(days=input_max_size_of_extremum)))
            if cr0 == 0:
                i+=1
                continue

            if prev_higher_px_date == datetime.min and input_min_size_of_extremum == 10000:
                cr1 = 1
            else:
                cr1 = int(1 * (dt[i] - prev_higher_px_date >= timedelta(days=input_min_size_of_extremum)))
                if prev_higher_px_date == datetime.min:
                    cr1 = int(1 * (dt[i] - dt[0] >= timedelta(days=input_min_size_of_extremum)))
            if cr1 == 0:
                i+=1
                continue


            time_of_min_idx = datetime.min
            prev_local_min = float('NaN')

            if idx_of_previously_higher_px != -1:
                prev_local_min  = min(px[idx_of_previously_higher_px:i])
                min_idx = px[idx_of_previously_higher_px:i].argmin() + idx_of_previously_higher_px #offset done
                time_of_min_idx = dt[min_idx] #dt[idx_of_previously_higher_px:i][min_idx] #
                prev_local_min = px[min_idx] #px[idx_of_previously_higher_px:i][min_idx]


            #SIGNAL TRIGGER
            if (cr0 + cr1 + cr2 + cr3) == 4:
                event_direction = 1
                if print_or_no == True:
                    print("-----------------------------------------------------------")
                    print("at i = ", i, " dt i = ", dt[i], " px = ", px[i])
                    print("idx of prev higher: ", idx_of_previously_higher_px)
                    print("px at prev higher point: ", prev_higher_px, " dt[px] = ", dt[idx_of_previously_higher_px])
                    print("at i = ", i, " px i = ", px[i], " dt[i] = ", dt[i])
                    print("diff t: ", dt[i] - prev_higher_px_date)
                    print("diff act: ", dt[i] - dt[0])
                    print("t: ", dt[i])
                    print("px: ", px[i])
                    print("max before: ", max(px[:i]))
                    print("dt 0: ", dt[0])
                    print("prev local min: ", prev_local_min)
                    print(" at time: ", time_of_min_idx)

				#ADDING AN EVENT / SIGNAL
				#record event_table
                event_dates.append(dt[i])
                event_prices_loc.append(px[i])
                event_tickers.append(input_settings[event.ticker_key])
                event_directions.append(input_trade_same_dir_as_extr_or_no * event_direction)

				#record calculation trail
                event_move_sizes_from_previous_extremum.append(px[i] / prev_local_min - 1)
                event_start_dates.append(prev_higher_px_date)
                number_of_days_extremum_over.append(((dt[i] - prev_higher_px_date).total_seconds()) / (60 * 1440))

                other_clue_params_slope.append((px[i]/prev_local_min-1)/((dt[i] - time_of_min_idx).total_seconds() / (60 * 1440)))
                other_clue_params_which_earlier.append(1)

                i+=1
                continue

		#SHORTS
        if input_mins_maxes_or_both == -1:
			...
			#***CUSTOMISE - add calculation here

		#***********CUSTOMISE: CALCULATION OF SIGNAL GOES HERE - END*******

        #########################################################
        i += 1 #loop counter incremented
    # while loop through each timestamp ends here


    cdef event_table = event.event_table(input_settings) #do not change

    #***********CUSTOMISE - START*******
    cdef calculation_trail = {
        "start dts":            event_start_dates,
        "move frm prev extr":   event_move_sizes_from_previous_extremum,
        'days extr over':       number_of_days_extremum_over,
		#ADD ANY CUSTOM TRAIL PARAMETERS HERE
    }
	#***********CUSTOMISE - END*******

    if len(event_start_dates) != len(event_move_sizes_from_previous_extremum): sys.exit("len diff")
    if len(event_start_dates) != len(number_of_days_extremum_over): sys.exit("len diff")

	#***********CUSTOMISE - START*******
    event_table.other_clue_parameters_trail = {
        'slope': other_clue_params_slope,
        'which earlier': other_clue_params_which_earlier,
		#ADD ANY CUSTOM ITEMS HERE
    }
	#***********CUSTOMISE - END*******

    #do not change
    event_table.event_dates = event_dates
    event_table.event_prices = event_prices_loc
    event_table.event_tickers = event_tickers
    event_table.event_directions = event_directions
    event_table.event_calculation_trail = calculation_trail

    if print_or_no: print("---Completed XXX calc in %s seconds ---" % (time.time() - start_time))

    return event_table