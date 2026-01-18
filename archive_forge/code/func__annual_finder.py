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
def _annual_finder(vmin, vmax, freq: BaseOffset) -> np.ndarray:
    vmin, vmax = (int(vmin), int(vmax + 1))
    span = vmax - vmin + 1
    info = np.zeros(span, dtype=[('val', int), ('maj', bool), ('min', bool), ('fmt', '|S8')])
    info['val'] = np.arange(vmin, vmax + 1)
    info['fmt'] = ''
    dates_ = info['val']
    min_anndef, maj_anndef = _get_default_annual_spacing(span)
    major_idx = dates_ % maj_anndef == 0
    minor_idx = dates_ % min_anndef == 0
    info['maj'][major_idx] = True
    info['min'][minor_idx] = True
    info['fmt'][major_idx] = '%Y'
    return info