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
def _from_ordinal(x, tz: tzinfo | None=None) -> datetime:
    ix = int(x)
    dt = datetime.fromordinal(ix)
    remainder = float(x) - ix
    hour, remainder = divmod(24 * remainder, 1)
    minute, remainder = divmod(60 * remainder, 1)
    second, remainder = divmod(60 * remainder, 1)
    microsecond = int(1000000 * remainder)
    if microsecond < 10:
        microsecond = 0
    dt = datetime(dt.year, dt.month, dt.day, int(hour), int(minute), int(second), microsecond)
    if tz is not None:
        dt = dt.astimezone(tz)
    if microsecond > 999990:
        dt += timedelta(microseconds=1000000 - microsecond)
    return dt