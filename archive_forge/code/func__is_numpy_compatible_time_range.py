from __future__ import annotations
import re
import warnings
from collections.abc import Hashable
from datetime import datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING, Callable, Union
import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime, OutOfBoundsTimedelta
from xarray.coding.variables import (
from xarray.core import indexing
from xarray.core.common import contains_cftime_datetimes, is_np_datetime_like
from xarray.core.duck_array_ops import asarray
from xarray.core.formatting import first_n_items, format_timestamp, last_item
from xarray.core.pdcompat import nanosecond_precision_timestamp
from xarray.core.utils import emit_user_level_warning
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import T_ChunkedArray, get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
from xarray.namedarray.utils import is_duck_dask_array
def _is_numpy_compatible_time_range(times):
    if is_np_datetime_like(times.dtype):
        return True
    times = np.asarray(times)
    tmin = times.min()
    tmax = times.max()
    try:
        convert_time_or_go_back(tmin, pd.Timestamp)
        convert_time_or_go_back(tmax, pd.Timestamp)
    except pd.errors.OutOfBoundsDatetime:
        return False
    except ValueError as err:
        if err.args[0] == 'year 0 is out of range':
            return False
        raise
    else:
        return True