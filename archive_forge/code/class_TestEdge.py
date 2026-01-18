import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
class TestEdge:

    def test_check_simple(self):
        a = np.arange(12)
        a = np.reshape(a, (4, 3))
        a = np.pad(a, ((2, 3), (3, 2)), 'edge')
        b = np.array([[0, 0, 0, 0, 1, 2, 2, 2], [0, 0, 0, 0, 1, 2, 2, 2], [0, 0, 0, 0, 1, 2, 2, 2], [3, 3, 3, 3, 4, 5, 5, 5], [6, 6, 6, 6, 7, 8, 8, 8], [9, 9, 9, 9, 10, 11, 11, 11], [9, 9, 9, 9, 10, 11, 11, 11], [9, 9, 9, 9, 10, 11, 11, 11], [9, 9, 9, 9, 10, 11, 11, 11]])
        assert_array_equal(a, b)

    def test_check_width_shape_1_2(self):
        a = np.array([1, 2, 3])
        padded = np.pad(a, ((1, 2),), 'edge')
        expected = np.array([1, 1, 2, 3, 3, 3])
        assert_array_equal(padded, expected)
        a = np.array([[1, 2, 3], [4, 5, 6]])
        padded = np.pad(a, ((1, 2),), 'edge')
        expected = np.pad(a, ((1, 2), (1, 2)), 'edge')
        assert_array_equal(padded, expected)
        a = np.arange(24).reshape(2, 3, 4)
        padded = np.pad(a, ((1, 2),), 'edge')
        expected = np.pad(a, ((1, 2), (1, 2), (1, 2)), 'edge')
        assert_array_equal(padded, expected)