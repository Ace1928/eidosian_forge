from __future__ import annotations
import warnings
from collections.abc import Hashable, MutableMapping
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Union
import numpy as np
import pandas as pd
from xarray.core import dtypes, duck_array_ops, indexing
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
def _choose_float_dtype(dtype: np.dtype, mapping: MutableMapping) -> type[np.floating[Any]]:
    """Return a float dtype that can losslessly represent `dtype` values."""
    scale_factor = mapping.get('scale_factor')
    add_offset = mapping.get('add_offset')
    if scale_factor is not None or add_offset is not None:
        if scale_factor is not None:
            scale_type = np.dtype(type(scale_factor))
        if add_offset is not None:
            offset_type = np.dtype(type(add_offset))
        if add_offset is not None and scale_factor is not None and (offset_type == scale_type) and (scale_type in [np.float32, np.float64]):
            if dtype.itemsize == 4 and np.issubdtype(dtype, np.integer):
                return np.float64
            return scale_type.type
        if add_offset is not None:
            return np.float64
        return scale_type.type
    if dtype.itemsize <= 4 and np.issubdtype(dtype, np.floating):
        return np.float32
    if dtype.itemsize <= 2 and np.issubdtype(dtype, np.integer):
        return np.float32
    return np.float64