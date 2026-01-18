from __future__ import annotations
import itertools
from collections.abc import Sequence
from functools import partial
from itertools import product
from numbers import Integral, Number
import numpy as np
from tlz import sliding_window
from dask.array import chunk
from dask.array.backends import array_creation_dispatch
from dask.array.core import (
from dask.array.numpy_compat import AxisError
from dask.array.ufunc import greater_equal, rint
from dask.array.utils import meta_from_array
from dask.array.wrap import empty, full, ones, zeros
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.utils import cached_cumsum, derived_from, is_cupy_type
def _get_like_function_shapes_chunks(a, chunks, shape):
    """
    Helper function for finding shapes and chunks for *_like()
    array creation functions.
    """
    if shape is None:
        shape = a.shape
        if chunks is None:
            chunks = a.chunks
    elif chunks is None:
        chunks = 'auto'
    return (shape, chunks)