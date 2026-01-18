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
def _numel_arraylike(x, **kwargs):
    """Numel implementation for arrays that want to return numel of the same type."""
    return _numel(x, coerce_np_ndarray=False, **kwargs)