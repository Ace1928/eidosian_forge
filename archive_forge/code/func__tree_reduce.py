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
def _tree_reduce(x, aggregate, axis, keepdims, dtype, split_every=None, combine=None, name=None, concatenate=True, reduced_meta=None):
    """Perform the tree reduction step of a reduction.

    Lower level, users should use ``reduction`` or ``arg_reduction`` directly.
    """
    split_every = split_every or config.get('split_every', 4)
    if isinstance(split_every, dict):
        split_every = {k: split_every.get(k, 2) for k in axis}
    elif isinstance(split_every, Integral):
        n = builtins.max(int(split_every ** (1 / (len(axis) or 1))), 2)
        split_every = dict.fromkeys(axis, n)
    else:
        raise ValueError('split_every must be a int or a dict')
    depth = 1
    for i, n in enumerate(x.numblocks):
        if i in split_every and split_every[i] != 1:
            depth = int(builtins.max(depth, math.ceil(math.log(n, split_every[i]))))
    func = partial(combine or aggregate, axis=axis, keepdims=True)
    if concatenate:
        func = compose(func, partial(_concatenate2, axes=sorted(axis)))
    for _ in range(depth - 1):
        x = partial_reduce(func, x, split_every, True, dtype=dtype, name=(name or funcname(combine or aggregate)) + '-partial', reduced_meta=reduced_meta)
    func = partial(aggregate, axis=axis, keepdims=keepdims)
    if concatenate:
        func = compose(func, partial(_concatenate2, axes=sorted(axis)))
    return partial_reduce(func, x, split_every, keepdims=keepdims, dtype=dtype, name=(name or funcname(aggregate)) + '-aggregate', reduced_meta=reduced_meta)