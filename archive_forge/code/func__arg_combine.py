from __future__ import annotations
import builtins
import contextlib
import math
import operator
import warnings
from collections.abc import Iterable
from functools import partial
from itertools import product, repeat
from numbers import Integral, Number
import numpy as np
from tlz import accumulate, compose, drop, get, partition_all, pluck
from dask import config
from dask.array import chunk
from dask.array.blockwise import blockwise
from dask.array.core import (
from dask.array.creation import arange, diagonal
from dask.array.dispatch import divide_lookup, nannumel_lookup, numel_lookup
from dask.array.numpy_compat import ComplexWarning
from dask.array.utils import (
from dask.array.wrap import ones, zeros
from dask.base import tokenize
from dask.blockwise import lol_tuples
from dask.highlevelgraph import HighLevelGraph
from dask.utils import (
def _arg_combine(data, axis, argfunc, keepdims=False):
    """Merge intermediate results from ``arg_*`` functions"""
    if isinstance(data, dict):
        assert data['vals'].ndim == data['arg'].ndim
        axis = None if len(axis) == data['vals'].ndim or data['vals'].ndim == 1 else axis[0]
    else:
        axis = None if len(axis) == data.ndim or data.ndim == 1 else axis[0]
    vals = data['vals']
    arg = data['arg']
    if axis is None:
        local_args = argfunc(vals, axis=axis, keepdims=keepdims)
        vals = vals.ravel()[local_args]
        arg = arg.ravel()[local_args]
    else:
        local_args = argfunc(vals, axis=axis)
        inds = list(np.ogrid[tuple(map(slice, local_args.shape))])
        inds.insert(axis, local_args)
        inds = tuple(inds)
        vals = vals[inds]
        arg = arg[inds]
        if keepdims:
            vals = np.expand_dims(vals, axis)
            arg = np.expand_dims(arg, axis)
    return (arg, vals)