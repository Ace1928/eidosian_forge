import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def _compare_index_result(self, arr, index, mimic_get, no_copy):
    """Compare mimicked result to indexing result.
        """
    arr = arr.copy()
    indexed_arr = arr[index]
    assert_array_equal(indexed_arr, mimic_get)
    if indexed_arr.size != 0 and indexed_arr.ndim != 0:
        assert_(np.may_share_memory(indexed_arr, arr) == no_copy)
        if HAS_REFCOUNT:
            if no_copy:
                assert_equal(sys.getrefcount(arr), 3)
            else:
                assert_equal(sys.getrefcount(arr), 2)
    b = arr.copy()
    b[index] = mimic_get + 1000
    if b.size == 0:
        return
    if no_copy and indexed_arr.ndim != 0:
        indexed_arr += 1000
        assert_array_equal(arr, b)
        return
    arr.flat[indexed_arr.ravel()] += 1000
    assert_array_equal(arr, b)