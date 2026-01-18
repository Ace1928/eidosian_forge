from __future__ import annotations
import math
import warnings
from collections.abc import Iterable
from functools import partial, reduce, wraps
from numbers import Integral, Real
import numpy as np
from tlz import concat, interleave, sliding_window
from dask.array import chunk
from dask.array.core import (
from dask.array.creation import arange, diag, empty, indices, tri
from dask.array.einsumfuncs import einsum  # noqa
from dask.array.numpy_compat import NUMPY_GE_200
from dask.array.reductions import reduction
from dask.array.ufunc import multiply, sqrt
from dask.array.utils import (
from dask.array.wrap import ones
from dask.base import is_dask_collection, tokenize
from dask.core import flatten
from dask.delayed import Delayed, unpack_collections
from dask.highlevelgraph import HighLevelGraph
from dask.utils import apply, derived_from, funcname, is_arraylike, is_cupy_type
def _average(a, axis=None, weights=None, returned=False, is_masked=False, keepdims=False):
    a = asanyarray(a)
    if weights is None:
        avg = a.mean(axis, keepdims=keepdims)
        scl = avg.dtype.type(a.size / avg.size)
    else:
        wgt = asanyarray(weights)
        if issubclass(a.dtype.type, (np.integer, np.bool_)):
            result_dtype = result_type(a.dtype, wgt.dtype, 'f8')
        else:
            result_dtype = result_type(a.dtype, wgt.dtype)
        if a.shape != wgt.shape:
            if axis is None:
                raise TypeError('Axis must be specified when shapes of a and weights differ.')
            if wgt.ndim != 1:
                raise TypeError('1D weights expected when shapes of a and weights differ.')
            if wgt.shape[0] != a.shape[axis]:
                raise ValueError('Length of weights not compatible with specified axis.')
            wgt = broadcast_to(wgt, (a.ndim - 1) * (1,) + wgt.shape)
            wgt = wgt.swapaxes(-1, axis)
        if is_masked:
            from dask.array.ma import getmaskarray
            wgt = wgt * ~getmaskarray(a)
        scl = wgt.sum(axis=axis, dtype=result_dtype, keepdims=keepdims)
        avg = multiply(a, wgt, dtype=result_dtype).sum(axis, keepdims=keepdims) / scl
    if returned:
        if scl.shape != avg.shape:
            scl = broadcast_to(scl, avg.shape).copy()
        return (avg, scl)
    else:
        return avg