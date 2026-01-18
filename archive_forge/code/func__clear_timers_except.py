import functools
import logging
import sys
import time
import traceback
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.modeling import NOTSET as _NotSpecified
def _clear_timers_except(timer, to_retain):
    """A helper function for removing keys, except for those specified,
    from the dictionary of timers

    Parameters
    ----------
    timer: HierarchicalTimer or _HierarchicalHelper
        The timer whose dict of "sub-timers" will be pruned

    to_retain: set
        Set of keys of the "sub-timers" to retain

    """
    keys = list(timer.timers.keys())
    for key in keys:
        if key not in to_retain:
            timer.timers.pop(key)