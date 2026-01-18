from __future__ import annotations
from collections.abc import Iterable
from functools import partial
from math import ceil
from operator import getitem
from threading import Lock
from typing import TYPE_CHECKING, Literal, overload
import numpy as np
import pandas as pd
import dask.array as da
from dask.base import is_dask_collection, tokenize
from dask.blockwise import BlockwiseDepDict, blockwise
from dask.dataframe._compat import is_any_real_numeric_dtype
from dask.dataframe.backends import dataframe_creation_dispatch
from dask.dataframe.core import (
from dask.dataframe.dispatch import meta_lib_from_array
from dask.dataframe.io.utils import DataFrameIOFunction
from dask.dataframe.utils import (
from dask.delayed import Delayed, delayed
from dask.highlevelgraph import HighLevelGraph
from dask.layers import DataFrameIOLayer
from dask.utils import M, funcname, is_arraylike
def _meta_from_array(x, columns=None, index=None, meta=None):
    """Create empty DataFrame or Series which has correct dtype"""
    if x.ndim > 2:
        raise ValueError('from_array does not input more than 2D array, got array with shape %r' % (x.shape,))
    if index is not None:
        if not isinstance(index, Index):
            raise ValueError("'index' must be an instance of dask.dataframe.Index")
        index = index._meta
    if meta is None:
        meta = meta_lib_from_array(x).DataFrame()
    if getattr(x.dtype, 'names', None) is not None:
        if columns is None:
            columns = list(x.dtype.names)
        elif np.isscalar(columns):
            raise ValueError('For a struct dtype, columns must be a list.')
        elif not all((i in x.dtype.names for i in columns)):
            extra = sorted(set(columns).difference(x.dtype.names))
            raise ValueError(f"dtype {x.dtype} doesn't have fields {extra}")
        fields = x.dtype.fields
        dtypes = [fields[n][0] if n in fields else 'f8' for n in columns]
    elif x.ndim == 1:
        if np.isscalar(columns) or columns is None:
            return meta._constructor_sliced([], name=columns, dtype=x.dtype, index=index)
        elif len(columns) == 1:
            return meta._constructor(np.array([], dtype=x.dtype), columns=columns, index=index)
        raise ValueError('For a 1d array, columns must be a scalar or single element list')
    else:
        if np.isnan(x.shape[1]):
            raise ValueError('Shape along axis 1 must be known')
        if columns is None:
            columns = list(range(x.shape[1])) if x.ndim == 2 else [0]
        elif len(columns) != x.shape[1]:
            raise ValueError(f'Number of column names must match width of the array. Got {len(columns)} names for {x.shape[1]} columns')
        dtypes = [x.dtype] * len(columns)
    data = {c: np.array([], dtype=dt) for c, dt in zip(columns, dtypes)}
    return meta._constructor(data, columns=columns, index=index)