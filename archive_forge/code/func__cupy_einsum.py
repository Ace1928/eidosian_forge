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
@einsum_lookup.register(cupy.ndarray)
def _cupy_einsum(*args, **kwargs):
    kwargs.pop('casting', None)
    kwargs.pop('order', None)
    return cupy.einsum(*args, **kwargs)