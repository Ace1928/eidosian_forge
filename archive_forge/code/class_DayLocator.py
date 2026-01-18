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
class DayLocator(RRuleLocator):
    """
    Make ticks on occurrences of each day of the month.  For example,
    1, 15, 30.
    """

    def __init__(self, bymonthday=None, interval=1, tz=None):
        """
        Parameters
        ----------
        bymonthday : int or list of int, default: all days
            Ticks will be placed on every day in *bymonthday*. Default is
            ``bymonthday=range(1, 32)``, i.e., every day of the month.
        interval : int, default: 1
            The interval between each iteration. For example, if
            ``interval=2``, mark every second occurrence.
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        if interval != int(interval) or interval < 1:
            raise ValueError('interval must be an integer greater than 0')
        if bymonthday is None:
            bymonthday = range(1, 32)
        rule = rrulewrapper(DAILY, bymonthday=bymonthday, interval=interval, **self.hms0d)
        super().__init__(rule, tz=tz)