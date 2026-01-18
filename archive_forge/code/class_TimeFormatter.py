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
class TimeFormatter(Formatter):

    def __init__(self, locs) -> None:
        self.locs = locs

    def __call__(self, x, pos: int | None=0) -> str:
        """
        Return the time of day as a formatted string.

        Parameters
        ----------
        x : float
            The time of day specified as seconds since 00:00 (midnight),
            with up to microsecond precision.
        pos
            Unused

        Returns
        -------
        str
            A string in HH:MM:SS.mmmuuu format. Microseconds,
            milliseconds and seconds are only displayed if non-zero.
        """
        fmt = '%H:%M:%S.%f'
        s = int(x)
        msus = round((x - s) * 10 ** 6)
        ms = msus // 1000
        us = msus % 1000
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        _, h = divmod(h, 24)
        if us != 0:
            return pydt.time(h, m, s, msus).strftime(fmt)
        elif ms != 0:
            return pydt.time(h, m, s, msus).strftime(fmt)[:-3]
        elif s != 0:
            return pydt.time(h, m, s).strftime('%H:%M:%S')
        return pydt.time(h, m).strftime('%H:%M')