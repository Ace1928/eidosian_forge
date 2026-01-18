from __future__ import annotations
import re
from functools import total_ordering
from packaging.version import Version
import numpy as np
import pandas as pd
from numba import jit
from pandas.api.extensions import (
from numbers import Integral
from pandas.api.types import pandas_dtype, is_extension_array_dtype
def _eq_ragged_ndarray1d(start_indices, flat_array, a):
    """
    Compare a RaggedArray with a 1D numpy object array of the same length

    Parameters
    ----------
    start_indices: ndarray
        start indices of a RaggedArray
    flat_array: ndarray
        flat_array property of a RaggedArray
    a: ndarray
        1D numpy array of same length as ra

    Returns
    -------
    mask: ndarray
        1D bool array of same length as input with elements True when
        corresponding elements are equal, False otherwise

    Notes
    -----
    This function is not numba accelerated because it, by design, inputs
    a numpy object array
    """
    n = len(start_indices)
    m = len(flat_array)
    result = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        start_index = start_indices[i]
        stop_index = start_indices[i + 1] if i < n - 1 else m
        a_val = a[i]
        if a_val is None or (np.isscalar(a_val) and np.isnan(a_val)) or len(a_val) == 0:
            result[i] = start_index == stop_index
        else:
            result[i] = np.array_equal(flat_array[start_index:stop_index], a_val)
    return result