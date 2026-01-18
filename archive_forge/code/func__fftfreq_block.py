from __future__ import annotations
import inspect
import warnings
from collections.abc import Sequence
import numpy as np
from dask.array.core import asarray
from dask.array.core import concatenate as _concatenate
from dask.array.creation import arange as _arange
from dask.array.numpy_compat import NUMPY_GE_200
from dask.utils import derived_from, skip_doctest
def _fftfreq_block(i, n, d):
    r = i.copy()
    r[i >= (n + 1) // 2] -= n
    r /= n * d
    return r