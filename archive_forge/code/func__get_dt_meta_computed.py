from __future__ import annotations
import contextlib
import functools
import itertools
import math
import numbers
import warnings
import numpy as np
from tlz import concat, frequencies
from dask.array.core import Array
from dask.array.numpy_compat import AxisError
from dask.base import is_dask_collection, tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.utils import has_keyword, is_arraylike, is_cupy_type, typename
def _get_dt_meta_computed(x, check_shape=True, check_graph=True, check_chunks=True, check_ndim=True, scheduler=None):
    x_original = x
    x_meta = None
    x_computed = None
    if is_dask_collection(x) and is_arraylike(x):
        assert x.dtype is not None
        adt = x.dtype
        if check_graph:
            _check_dsk(x.dask)
        x_meta = getattr(x, '_meta', None)
        if check_chunks:
            x = _check_chunks(x, check_ndim=check_ndim, scheduler=scheduler)
        x = x.compute(scheduler=scheduler)
        x_computed = x
        if hasattr(x, 'todense'):
            x = x.todense()
        if not hasattr(x, 'dtype'):
            x = np.array(x, dtype='O')
        if _not_empty(x):
            assert x.dtype == x_original.dtype
        if check_shape:
            assert_eq_shape(x_original.shape, x.shape, check_nan=False)
    else:
        if not hasattr(x, 'dtype'):
            x = np.array(x, dtype='O')
        adt = getattr(x, 'dtype', None)
    return (x, adt, x_meta, x_computed)