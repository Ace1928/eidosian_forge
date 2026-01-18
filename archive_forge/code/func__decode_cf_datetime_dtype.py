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
def _decode_cf_datetime_dtype(data, units: str, calendar: str, use_cftime: bool | None) -> np.dtype:
    values = indexing.ImplicitToExplicitIndexingAdapter(indexing.as_indexable(data))
    example_value = np.concatenate([first_n_items(values, 1) or [0], last_item(values) or [0]])
    try:
        result = decode_cf_datetime(example_value, units, calendar, use_cftime)
    except Exception:
        calendar_msg = 'the default calendar' if calendar is None else f'calendar {calendar!r}'
        msg = f'unable to decode time units {units!r} with {calendar_msg!r}. Try opening your dataset with decode_times=False or installing cftime if it is not installed.'
        raise ValueError(msg)
    else:
        dtype = getattr(result, 'dtype', np.dtype('object'))
    return dtype