import sys
import random
import datetime
import simplejson as json
from collections import OrderedDict

    Generates a gantt chart in html showing the workflow execution based on a callback log file.
    This script was intended to be used with the MultiprocPlugin.
    The following code shows how to set up the workflow in order to generate the log file:

    Parameters
    ----------
    logfile : string
        filepath to the callback log file to plot the gantt chart of
    cores : integer
        the number of cores given to the workflow via the 'n_procs'
        plugin arg
    minute_scale : integer (optional); default=10
        the scale, in minutes, at which to plot line markers for the
        gantt chart; for example, minute_scale=10 means there are lines
        drawn at every 10 minute interval from start to finish
    space_between_minutes : integer (optional); default=50
        scale factor in pixel spacing between minute line markers
    colors : list (optional)
        a list of colors to choose from when coloring the nodes in the
        gantt chart


    Returns
    -------
    None
        the function does not return any value but writes out an html
        file in the same directory as the callback log path passed in

    Usage
    -----
    # import logging
    # import logging.handlers
    # from nipype.utils.profiler import log_nodes_cb

    # log_filename = 'callback.log'
    # logger = logging.getLogger('callback')
    # logger.setLevel(logging.DEBUG)
    # handler = logging.FileHandler(log_filename)
    # logger.addHandler(handler)

    # #create workflow
    # workflow = ...

    # workflow.run(plugin='MultiProc',
    #     plugin_args={'n_procs':8, 'memory':12, 'status_callback': log_nodes_cb})

    # generate_gantt_chart('callback.log', 8)
    