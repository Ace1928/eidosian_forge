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
def _get_date_field(values, name, dtype):
    """Indirectly access pandas' libts.get_date_field by wrapping data
    as a Series and calling through `.dt` attribute.

    Parameters
    ----------
    values : np.ndarray or dask.array-like
        Array-like container of datetime-like values
    name : str
        Name of datetime field to access
    dtype : dtype-like
        dtype for output date field values

    Returns
    -------
    datetime_fields : same type as values
        Array-like of datetime fields accessed for each element in values

    """
    if is_np_datetime_like(values.dtype):
        access_method = _access_through_series
    else:
        access_method = _access_through_cftimeindex
    if is_duck_dask_array(values):
        from dask.array import map_blocks
        new_axis = chunks = None
        if name == 'isocalendar':
            chunks = (3,) + values.chunksize
            new_axis = 0
        return map_blocks(access_method, values, name, dtype=dtype, new_axis=new_axis, chunks=chunks)
    else:
        out = access_method(values, name)
        if np.issubdtype(out.dtype, np.integer):
            out = out.astype(dtype, copy=False)
        return out