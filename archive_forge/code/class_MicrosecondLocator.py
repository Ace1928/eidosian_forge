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
class MicrosecondLocator(DateLocator):
    """
    Make ticks on regular intervals of one or more microsecond(s).

    .. note::

        By default, Matplotlib uses a floating point representation of time in
        days since the epoch, so plotting data with
        microsecond time resolution does not work well for
        dates that are far (about 70 years) from the epoch (check with
        `~.dates.get_epoch`).

        If you want sub-microsecond resolution time plots, it is strongly
        recommended to use floating point seconds, not datetime-like
        time representation.

        If you really must use datetime.datetime() or similar and still
        need microsecond precision, change the time origin via
        `.dates.set_epoch` to something closer to the dates being plotted.
        See :doc:`/gallery/ticks/date_precision_and_epochs`.

    """

    def __init__(self, interval=1, tz=None):
        """
        Parameters
        ----------
        interval : int, default: 1
            The interval between each iteration. For example, if
            ``interval=2``, mark every second occurrence.
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        super().__init__(tz=tz)
        self._interval = interval
        self._wrapped_locator = ticker.MultipleLocator(interval)

    def set_axis(self, axis):
        self._wrapped_locator.set_axis(axis)
        return super().set_axis(axis)

    def __call__(self):
        try:
            dmin, dmax = self.viewlim_to_dt()
        except ValueError:
            return []
        return self.tick_values(dmin, dmax)

    def tick_values(self, vmin, vmax):
        nmin, nmax = date2num((vmin, vmax))
        t0 = np.floor(nmin)
        nmax = nmax - t0
        nmin = nmin - t0
        nmin *= MUSECONDS_PER_DAY
        nmax *= MUSECONDS_PER_DAY
        ticks = self._wrapped_locator.tick_values(nmin, nmax)
        ticks = ticks / MUSECONDS_PER_DAY + t0
        return ticks

    def _get_unit(self):
        return 1.0 / MUSECONDS_PER_DAY

    def _get_interval(self):
        return self._interval