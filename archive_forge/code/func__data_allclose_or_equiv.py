import functools
import warnings
from collections.abc import Hashable
from typing import Union
import numpy as np
import pandas as pd
from xarray.core import duck_array_ops, formatting, utils
from xarray.core.coordinates import Coordinates
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.indexes import Index, PandasIndex, PandasMultiIndex, default_indexes
from xarray.core.variable import IndexVariable, Variable
def _data_allclose_or_equiv(arr1, arr2, rtol=1e-05, atol=1e-08, decode_bytes=True):
    if any((arr.dtype.kind == 'S' for arr in [arr1, arr2])) and decode_bytes:
        arr1 = _decode_string_data(arr1)
        arr2 = _decode_string_data(arr2)
    exact_dtypes = ['M', 'm', 'O', 'S', 'U']
    if any((arr.dtype.kind in exact_dtypes for arr in [arr1, arr2])):
        return duck_array_ops.array_equiv(arr1, arr2)
    else:
        return duck_array_ops.allclose_or_equiv(arr1, arr2, rtol=rtol, atol=atol)