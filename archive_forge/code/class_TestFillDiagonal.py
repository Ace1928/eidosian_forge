import pytest
import numpy as np
from numpy.testing import (
from numpy.lib.index_tricks import (
class TestFillDiagonal:

    def test_basic(self):
        a = np.zeros((3, 3), int)
        fill_diagonal(a, 5)
        assert_array_equal(a, np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]]))

    def test_tall_matrix(self):
        a = np.zeros((10, 3), int)
        fill_diagonal(a, 5)
        assert_array_equal(a, np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]))

    def test_tall_matrix_wrap(self):
        a = np.zeros((10, 3), int)
        fill_diagonal(a, 5, True)
        assert_array_equal(a, np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5], [0, 0, 0], [5, 0, 0], [0, 5, 0], [0, 0, 5], [0, 0, 0], [5, 0, 0], [0, 5, 0]]))

    def test_wide_matrix(self):
        a = np.zeros((3, 10), int)
        fill_diagonal(a, 5)
        assert_array_equal(a, np.array([[5, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 5, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 5, 0, 0, 0, 0, 0, 0, 0]]))

    def test_operate_4d_array(self):
        a = np.zeros((3, 3, 3, 3), int)
        fill_diagonal(a, 4)
        i = np.array([0, 1, 2])
        assert_equal(np.where(a != 0), (i, i, i, i))

    def test_low_dim_handling(self):
        a = np.zeros(3, int)
        with assert_raises_regex(ValueError, 'at least 2-d'):
            fill_diagonal(a, 5)

    def test_hetero_shape_handling(self):
        a = np.zeros((3, 3, 7, 3), int)
        with assert_raises_regex(ValueError, 'equal length'):
            fill_diagonal(a, 2)