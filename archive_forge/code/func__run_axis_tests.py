import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
def _run_axis_tests(self, dtype):
    data = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]]).astype(dtype)
    msg = 'Unique with 1d array and axis=0 failed'
    result = np.array([0, 1])
    assert_array_equal(unique(data), result.astype(dtype), msg)
    msg = 'Unique with 2d array and axis=0 failed'
    result = np.array([[0, 1, 0, 0], [1, 0, 0, 0]])
    assert_array_equal(unique(data, axis=0), result.astype(dtype), msg)
    msg = 'Unique with 2d array and axis=1 failed'
    result = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
    assert_array_equal(unique(data, axis=1), result.astype(dtype), msg)
    msg = 'Unique with 3d array and axis=2 failed'
    data3d = np.array([[[1, 1], [1, 0]], [[0, 1], [0, 0]]]).astype(dtype)
    result = np.take(data3d, [1, 0], axis=2)
    assert_array_equal(unique(data3d, axis=2), result, msg)
    uniq, idx, inv, cnt = unique(data, axis=0, return_index=True, return_inverse=True, return_counts=True)
    msg = "Unique's return_index=True failed with axis=0"
    assert_array_equal(data[idx], uniq, msg)
    msg = "Unique's return_inverse=True failed with axis=0"
    assert_array_equal(uniq[inv], data)
    msg = "Unique's return_counts=True failed with axis=0"
    assert_array_equal(cnt, np.array([2, 2]), msg)
    uniq, idx, inv, cnt = unique(data, axis=1, return_index=True, return_inverse=True, return_counts=True)
    msg = "Unique's return_index=True failed with axis=1"
    assert_array_equal(data[:, idx], uniq)
    msg = "Unique's return_inverse=True failed with axis=1"
    assert_array_equal(uniq[:, inv], data)
    msg = "Unique's return_counts=True failed with axis=1"
    assert_array_equal(cnt, np.array([2, 1, 1]), msg)