from numpy.testing import (
from numpy import (
import numpy as np
import pytest
class TestTriuIndices:

    def test_triu_indices(self):
        iu1 = triu_indices(4)
        iu2 = triu_indices(4, k=2)
        iu3 = triu_indices(4, m=5)
        iu4 = triu_indices(4, k=2, m=5)
        a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        b = np.arange(1, 21).reshape(4, 5)
        assert_array_equal(a[iu1], array([1, 2, 3, 4, 6, 7, 8, 11, 12, 16]))
        assert_array_equal(b[iu3], array([1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14, 15, 19, 20]))
        a[iu1] = -1
        assert_array_equal(a, array([[-1, -1, -1, -1], [5, -1, -1, -1], [9, 10, -1, -1], [13, 14, 15, -1]]))
        b[iu3] = -1
        assert_array_equal(b, array([[-1, -1, -1, -1, -1], [6, -1, -1, -1, -1], [11, 12, -1, -1, -1], [16, 17, 18, -1, -1]]))
        a[iu2] = -10
        assert_array_equal(a, array([[-1, -1, -10, -10], [5, -1, -1, -10], [9, 10, -1, -1], [13, 14, 15, -1]]))
        b[iu4] = -10
        assert_array_equal(b, array([[-1, -1, -10, -10, -10], [6, -1, -1, -10, -10], [11, 12, -1, -1, -10], [16, 17, 18, -1, -1]]))