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
def _eq_ragged_ragged(start_indices1, flat_array1, start_indices2, flat_array2):
    """
    Compare elements of two ragged arrays of the same length

    Parameters
    ----------
    start_indices1: ndarray
        start indices of a RaggedArray 1
    flat_array1: ndarray
        flat_array property of a RaggedArray 1
    start_indices2: ndarray
        start indices of a RaggedArray 2
    flat_array2: ndarray
        flat_array property of a RaggedArray 2

    Returns
    -------
    mask: ndarray
        1D bool array of same length as inputs with elements True when
        corresponding elements are equal, False otherwise
    """
    n = len(start_indices1)
    m1 = len(flat_array1)
    m2 = len(flat_array2)
    result = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        start_index1 = start_indices1[i]
        stop_index1 = start_indices1[i + 1] if i < n - 1 else m1
        len_1 = stop_index1 - start_index1
        start_index2 = start_indices2[i]
        stop_index2 = start_indices2[i + 1] if i < n - 1 else m2
        len_2 = stop_index2 - start_index2
        if len_1 != len_2:
            el_equal = False
        else:
            el_equal = True
            for flat_index1, flat_index2 in zip(range(start_index1, stop_index1), range(start_index2, stop_index2)):
                el_1 = flat_array1[flat_index1]
                el_2 = flat_array2[flat_index2]
                el_equal &= el_1 == el_2
        result[i] = el_equal
    return result