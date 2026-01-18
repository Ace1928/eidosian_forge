from __future__ import annotations
import math
import numpy as np
from dask.array import chunk
from dask.array.core import Array
from dask.array.dispatch import (
from dask.array.numpy_compat import divide as np_divide
from dask.array.numpy_compat import ma_divide
from dask.array.percentile import _percentile
from dask.backends import CreationDispatch, DaskBackendEntrypoint
def _concat_cupy_sparse(L, axis=0):
    if axis == 0:
        return vstack(L)
    elif axis == 1:
        return hstack(L)
    else:
        msg = 'Can only concatenate cupy sparse matrices for axis in {0, 1}.  Got %s' % axis
        raise ValueError(msg)