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
def _get_periods_per_ymd(freq: BaseOffset) -> tuple[int, int, int]:
    dtype_code = freq._period_dtype_code
    freq_group = FreqGroup.from_period_dtype_code(dtype_code)
    ppd = -1
    if dtype_code >= FreqGroup.FR_HR.value:
        ppd = periods_per_day(freq._creso)
        ppm = 28 * ppd
        ppy = 365 * ppd
    elif freq_group == FreqGroup.FR_BUS:
        ppm = 19
        ppy = 261
    elif freq_group == FreqGroup.FR_DAY:
        ppm = 28
        ppy = 365
    elif freq_group == FreqGroup.FR_WK:
        ppm = 3
        ppy = 52
    elif freq_group == FreqGroup.FR_MTH:
        ppm = 1
        ppy = 12
    elif freq_group == FreqGroup.FR_QTR:
        ppm = -1
        ppy = 4
    elif freq_group == FreqGroup.FR_ANN:
        ppm = -1
        ppy = 1
    else:
        raise NotImplementedError(f'Unsupported frequency: {dtype_code}')
    return (ppd, ppm, ppy)