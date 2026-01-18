from __future__ import annotations
import contextlib
import datetime as pydt
from datetime import (
import functools
from typing import (
import warnings
import matplotlib.dates as mdates
from matplotlib.ticker import (
from matplotlib.transforms import nonsingular
import matplotlib.units as munits
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import (
from pandas._typing import (
from pandas.core.dtypes.common import (
from pandas import (
import pandas.core.common as com
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
import pandas.core.tools.datetimes as tools
class MilliSecondLocator(mdates.DateLocator):
    UNIT = 1.0 / (24 * 3600 * 1000)

    def __init__(self, tz) -> None:
        mdates.DateLocator.__init__(self, tz)
        self._interval = 1.0

    def _get_unit(self):
        return self.get_unit_generic(-1)

    @staticmethod
    def get_unit_generic(freq):
        unit = mdates.RRuleLocator.get_unit_generic(freq)
        if unit < 0:
            return MilliSecondLocator.UNIT
        return unit

    def __call__(self):
        try:
            dmin, dmax = self.viewlim_to_dt()
        except ValueError:
            return []
        nmax, nmin = mdates.date2num((dmax, dmin))
        num = (nmax - nmin) * 86400 * 1000
        max_millis_ticks = 6
        for interval in [1, 10, 50, 100, 200, 500]:
            if num <= interval * (max_millis_ticks - 1):
                self._interval = interval
                break
            self._interval = 1000.0
        estimate = (nmax - nmin) / (self._get_unit() * self._get_interval())
        if estimate > self.MAXTICKS * 2:
            raise RuntimeError(f'MillisecondLocator estimated to generate {estimate:d} ticks from {dmin} to {dmax}: exceeds Locator.MAXTICKS* 2 ({self.MAXTICKS * 2:d}) ')
        interval = self._get_interval()
        freq = f'{interval}ms'
        tz = self.tz.tzname(None)
        st = dmin.replace(tzinfo=None)
        ed = dmin.replace(tzinfo=None)
        all_dates = date_range(start=st, end=ed, freq=freq, tz=tz).astype(object)
        try:
            if len(all_dates) > 0:
                locs = self.raise_if_exceeds(mdates.date2num(all_dates))
                return locs
        except Exception:
            pass
        lims = mdates.date2num([dmin, dmax])
        return lims

    def _get_interval(self):
        return self._interval

    def autoscale(self):
        """
        Set the view limits to include the data range.
        """
        dmin, dmax = self.datalim_to_dt()
        vmin = mdates.date2num(dmin)
        vmax = mdates.date2num(dmax)
        return self.nonsingular(vmin, vmax)