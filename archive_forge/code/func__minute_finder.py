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
def _minute_finder(label_interval: int) -> None:
    target = dates_.minute
    hour_start = _period_break(dates_, 'hour')
    mask = _period_break_mask(dates_, 'minute')
    info_maj[hour_start] = True
    info_min[mask & (target % label_interval == 0)] = True
    info_fmt[mask & (target % label_interval == 0)] = '%H:%M'
    info_fmt[day_start] = '%H:%M\n%d-%b'
    info_fmt[year_start] = '%H:%M\n%d-%b\n%Y'