from __future__ import annotations
import warnings
import numpy as np
from xarray.core import dtypes, duck_array_ops, nputils, utils
from xarray.core.duck_array_ops import (
def _nan_minmax_object(func, fill_value, value, axis=None, **kwargs):
    """In house nanmin and nanmax for object array"""
    valid_count = count(value, axis=axis)
    filled_value = fillna(value, fill_value)
    data = getattr(np, func)(filled_value, axis=axis, **kwargs)
    if not hasattr(data, 'dtype'):
        data = fill_value if valid_count == 0 else data
        return utils.to_0d_object_array(data)
    return where_method(data, valid_count != 0)