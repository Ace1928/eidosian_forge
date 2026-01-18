from __future__ import annotations
from itertools import groupby
from math import floor, ceil
import dask.array as da
import numpy as np
from dask.delayed import delayed
from numba import prange
from .utils import ngjit, ngjit_parallel
def _get_out(out, src, shape):
    if out is None:
        return np.zeros(shape, dtype=src.dtype)
    else:
        if out.shape != shape:
            raise ValueError("'shape' and 'out' are incompatible")
        if out.shape == src.shape:
            return None
        return out