import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
class TestColumnStack:

    def test_non_iterable(self):
        assert_raises(TypeError, column_stack, 1)

    def test_1D_arrays(self):
        a = np.array((1, 2, 3))
        b = np.array((2, 3, 4))
        expected = np.array([[1, 2], [2, 3], [3, 4]])
        actual = np.column_stack((a, b))
        assert_equal(actual, expected)

    def test_2D_arrays(self):
        a = np.array([[1], [2], [3]])
        b = np.array([[2], [3], [4]])
        expected = np.array([[1, 2], [2, 3], [3, 4]])
        actual = np.column_stack((a, b))
        assert_equal(actual, expected)

    def test_generator(self):
        with pytest.raises(TypeError, match='arrays to stack must be'):
            column_stack((np.arange(3) for _ in range(2)))