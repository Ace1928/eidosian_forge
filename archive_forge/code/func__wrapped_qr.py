from __future__ import annotations
import operator
import warnings
from functools import partial
from numbers import Number
import numpy as np
import tlz as toolz
from dask.array.core import Array, concatenate, dotmany, from_delayed
from dask.array.creation import eye
from dask.array.random import RandomState, default_rng
from dask.array.utils import (
from dask.base import tokenize, wait
from dask.blockwise import blockwise
from dask.delayed import delayed
from dask.highlevelgraph import HighLevelGraph
from dask.utils import apply, derived_from
def _wrapped_qr(a):
    """
    A wrapper for np.linalg.qr that handles arrays with 0 rows

    Notes: Created for tsqr so as to manage cases with uncertain
    array dimensions. In particular, the case where arrays have
    (uncertain) chunks with 0 rows.
    """
    if a.shape[0] == 0:
        return (np.zeros_like(a, shape=(0, 0)), np.zeros_like(a, shape=(0, a.shape[1])))
    else:
        return np.linalg.qr(a)