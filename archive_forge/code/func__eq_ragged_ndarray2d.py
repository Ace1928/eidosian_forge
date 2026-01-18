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
@jit(nopython=True, nogil=True)
def _eq_ragged_ndarray2d(start_indices, flat_array, a):
    """
    Compare a RaggedArray with rows of a 2D numpy object array

    Parameters
    ----------
    start_indices: ndarray
        start indices of a RaggedArray
    flat_array: ndarray
        flat_array property of a RaggedArray
    a: ndarray
        A 2D numpy array where the length of the first dimension matches the
        length of the RaggedArray

    Returns
    -------
    mask: ndarray
        1D bool array of same length as input RaggedArray with elements True
        when corresponding elements of ra equal corresponding row of `a`
    """
    n = len(start_indices)
    m = len(flat_array)
    cols = a.shape[1]
    result = np.zeros(n, dtype=np.bool_)
    for row in range(n):
        start_index = start_indices[row]
        stop_index = start_indices[row + 1] if row < n - 1 else m
        if stop_index - start_index != cols:
            el_equal = False
        else:
            el_equal = True
            for col, flat_index in enumerate(range(start_index, stop_index)):
                el_equal &= flat_array[flat_index] == a[row, col]
        result[row] = el_equal
    return result