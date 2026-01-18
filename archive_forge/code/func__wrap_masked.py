from __future__ import annotations
from functools import wraps
import numpy as np
from dask.array import chunk
from dask.array.core import asanyarray, blockwise, elemwise, map_blocks
from dask.array.reductions import reduction
from dask.array.routines import _average
from dask.array.routines import nonzero as _nonzero
from dask.base import normalize_token
from dask.utils import derived_from
def _wrap_masked(f):

    @wraps(f)
    def _(a, value):
        a = asanyarray(a)
        value = asanyarray(value)
        ainds = tuple(range(a.ndim))[::-1]
        vinds = tuple(range(value.ndim))[::-1]
        oinds = max(ainds, vinds, key=len)
        return blockwise(f, oinds, a, ainds, value, vinds, dtype=a.dtype)
    return _