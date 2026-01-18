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
def _prefixscan_combine(func, binop, pre, x, axis, dtype):
    """Combine results of a parallel prefix scan such as cumsum

    Parameters
    ----------
    func : callable
        Cumulative function (e.g. ``np.cumsum``)
    binop : callable
        Associative function (e.g. ``add``)
    pre : np.array
        The value calculated in parallel from ``preop``.
        For example, the sum of all the previous blocks.
    x : np.array
        Current block
    axis : int
    dtype : dtype

    Returns
    -------
    np.array
    """
    return binop(pre, func(x, axis=axis, dtype=dtype))