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
class YearLocator(RRuleLocator):
    """
    Make ticks on a given day of each year that is a multiple of base.

    Examples::

      # Tick every year on Jan 1st
      locator = YearLocator()

      # Tick every 5 years on July 4th
      locator = YearLocator(5, month=7, day=4)
    """

    def __init__(self, base=1, month=1, day=1, tz=None):
        """
        Parameters
        ----------
        base : int, default: 1
            Mark ticks every *base* years.
        month : int, default: 1
            The month on which to place the ticks, starting from 1. Default is
            January.
        day : int, default: 1
            The day on which to place the ticks.
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        rule = rrulewrapper(YEARLY, interval=base, bymonth=month, bymonthday=day, **self.hms0d)
        super().__init__(rule, tz=tz)
        self.base = ticker._Edge_integer(base, 0)

    def _create_rrule(self, vmin, vmax):
        ymin = max(self.base.le(vmin.year) * self.base.step, 1)
        ymax = min(self.base.ge(vmax.year) * self.base.step, 9999)
        c = self.rule._construct
        replace = {'year': ymin, 'month': c.get('bymonth', 1), 'day': c.get('bymonthday', 1), 'hour': 0, 'minute': 0, 'second': 0}
        start = vmin.replace(**replace)
        stop = start.replace(year=ymax)
        self.rule.set(dtstart=start, until=stop)
        return (start, stop)