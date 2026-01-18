from __future__ import annotations
import warnings
from typing import TYPE_CHECKING, Generic
import numpy as np
import pandas as pd
from xarray.coding.times import infer_calendar_name
from xarray.core import duck_array_ops
from xarray.core.common import (
from xarray.core.types import T_DataArray
from xarray.core.variable import IndexVariable
from xarray.namedarray.utils import is_duck_dask_array
def _round_through_series_or_index(values, name, freq):
    """Coerce an array of datetime-like values to a pandas Series or xarray
    CFTimeIndex and apply requested rounding
    """
    from xarray.coding.cftimeindex import CFTimeIndex
    if is_np_datetime_like(values.dtype):
        values_as_series = pd.Series(duck_array_ops.ravel(values), copy=False)
        method = getattr(values_as_series.dt, name)
    else:
        values_as_cftimeindex = CFTimeIndex(duck_array_ops.ravel(values))
        method = getattr(values_as_cftimeindex, name)
    field_values = method(freq=freq).values
    return field_values.reshape(values.shape)