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
def _hfft_out_chunks(a, s, axes):
    assert len(axes) == 1
    axis = axes[0]
    if s is None:
        s = [2 * (a.chunks[axis][0] - 1)]
    n = s[0]
    chunks = list(a.chunks)
    chunks[axis] = (n,)
    return chunks