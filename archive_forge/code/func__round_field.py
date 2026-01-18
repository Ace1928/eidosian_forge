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
def _round_field(values, name, freq):
    """Indirectly access rounding functions by wrapping data
    as a Series or CFTimeIndex

    Parameters
    ----------
    values : np.ndarray or dask.array-like
        Array-like container of datetime-like values
    name : {"ceil", "floor", "round"}
        Name of rounding function
    freq : str
        a freq string indicating the rounding resolution

    Returns
    -------
    rounded timestamps : same type as values
        Array-like of datetime fields accessed for each element in values

    """
    if is_duck_dask_array(values):
        from dask.array import map_blocks
        dtype = np.datetime64 if is_np_datetime_like(values.dtype) else np.dtype('O')
        return map_blocks(_round_through_series_or_index, values, name, freq=freq, dtype=dtype)
    else:
        return _round_through_series_or_index(values, name, freq)