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
class TimeSeries_DateLocator(Locator):
    """
    Locates the ticks along an axis controlled by a :class:`Series`.

    Parameters
    ----------
    freq : BaseOffset
        Valid frequency specifier.
    minor_locator : {False, True}, optional
        Whether the locator is for minor ticks (True) or not.
    dynamic_mode : {True, False}, optional
        Whether the locator should work in dynamic mode.
    base : {int}, optional
    quarter : {int}, optional
    month : {int}, optional
    day : {int}, optional
    """
    axis: Axis

    def __init__(self, freq: BaseOffset, minor_locator: bool=False, dynamic_mode: bool=True, base: int=1, quarter: int=1, month: int=1, day: int=1, plot_obj=None) -> None:
        freq = to_offset(freq, is_period=True)
        self.freq = freq
        self.base = base
        self.quarter, self.month, self.day = (quarter, month, day)
        self.isminor = minor_locator
        self.isdynamic = dynamic_mode
        self.offset = 0
        self.plot_obj = plot_obj
        self.finder = get_finder(freq)

    def _get_default_locs(self, vmin, vmax):
        """Returns the default locations of ticks."""
        locator = self.finder(vmin, vmax, self.freq)
        if self.isminor:
            return np.compress(locator['min'], locator['val'])
        return np.compress(locator['maj'], locator['val'])

    def __call__(self):
        """Return the locations of the ticks."""
        vi = tuple(self.axis.get_view_interval())
        vmin, vmax = vi
        if vmax < vmin:
            vmin, vmax = (vmax, vmin)
        if self.isdynamic:
            locs = self._get_default_locs(vmin, vmax)
        else:
            base = self.base
            d, m = divmod(vmin, base)
            vmin = (d + 1) * base
            locs = list(range(vmin, vmax + 1, base))
        return locs

    def autoscale(self):
        """
        Sets the view limits to the nearest multiples of base that contain the
        data.
        """
        vmin, vmax = self.axis.get_data_interval()
        locs = self._get_default_locs(vmin, vmax)
        vmin, vmax = locs[[0, -1]]
        if vmin == vmax:
            vmin -= 1
            vmax += 1
        return nonsingular(vmin, vmax)