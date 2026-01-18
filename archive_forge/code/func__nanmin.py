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
def _nanmin(m, n):
    k_0 = min([m, n])
    k_1 = m if np.isnan(n) else n
    return k_1 if np.isnan(k_0) else k_0