import datetime
import functools
import logging
import re
from dateutil.rrule import (rrule, MO, TU, WE, TH, FR, SA, SU, YEARLY,
from dateutil.relativedelta import relativedelta
import dateutil.parser
import dateutil.tz
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, ticker, units
class HourLocator(RRuleLocator):
    """
    Make ticks on occurrences of each hour.
    """

    def __init__(self, byhour=None, interval=1, tz=None):
        """
        Parameters
        ----------
        byhour : int or list of int, default: all hours
            Ticks will be placed on every hour in *byhour*. Default is
            ``byhour=range(24)``, i.e., every hour.
        interval : int, default: 1
            The interval between each iteration. For example, if
            ``interval=2``, mark every second occurrence.
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        if byhour is None:
            byhour = range(24)
        rule = rrulewrapper(HOURLY, byhour=byhour, interval=interval, byminute=0, bysecond=0)
        super().__init__(rule, tz=tz)