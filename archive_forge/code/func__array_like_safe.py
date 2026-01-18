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
def _array_like_safe(np_func, da_func, a, like, **kwargs):
    if like is a and hasattr(a, '__array_function__'):
        return a
    if isinstance(like, Array):
        return da_func(a, **kwargs)
    elif isinstance(a, Array):
        if is_cupy_type(a._meta):
            a = a.compute(scheduler='sync')
    try:
        return np_func(a, like=meta_from_array(like), **kwargs)
    except TypeError:
        return np_func(a, **kwargs)