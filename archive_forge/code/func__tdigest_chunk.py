from __future__ import annotations
import warnings
from collections.abc import Iterator
from functools import wraps
from numbers import Number
import numpy as np
from tlz import merge
from dask.array.core import Array
from dask.array.numpy_compat import NUMPY_GE_122
from dask.array.numpy_compat import percentile as np_percentile
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
def _tdigest_chunk(a):
    from crick import TDigest
    t = TDigest()
    t.update(a)
    return t