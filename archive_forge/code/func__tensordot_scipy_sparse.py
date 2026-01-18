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
def _tensordot_scipy_sparse(a, b, axes):
    assert a.ndim == b.ndim == 2
    assert len(axes[0]) == len(axes[1]) == 1
    a_axis, = axes[0]
    b_axis, = axes[1]
    assert a_axis in (0, 1) and b_axis in (0, 1)
    assert a.shape[a_axis] == b.shape[b_axis]
    if a_axis == 0 and b_axis == 0:
        return a.T * b
    elif a_axis == 0 and b_axis == 1:
        return a.T * b.T
    elif a_axis == 1 and b_axis == 0:
        return a * b
    elif a_axis == 1 and b_axis == 1:
        return a * b.T