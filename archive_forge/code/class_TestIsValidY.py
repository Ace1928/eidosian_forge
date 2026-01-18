import sys
import os.path
from functools import wraps, partial
import weakref
import numpy as np
import warnings
from numpy.linalg import norm
from numpy.testing import (verbose, assert_,
import pytest
import scipy.spatial.distance
from scipy.spatial.distance import (
from scipy.spatial.distance import (braycurtis, canberra, chebyshev, cityblock,
from scipy._lib._util import np_long, np_ulong
class TestIsValidY:

    def test_is_valid_y_improper_shape_2D_E(self):
        y = np.zeros((3, 3), dtype=np.float64)
        with pytest.raises(ValueError):
            is_valid_y_throw(y)

    def test_is_valid_y_improper_shape_2D_F(self):
        y = np.zeros((3, 3), dtype=np.float64)
        assert_equal(is_valid_y(y), False)

    def test_is_valid_y_improper_shape_3D_E(self):
        y = np.zeros((3, 3, 3), dtype=np.float64)
        with pytest.raises(ValueError):
            is_valid_y_throw(y)

    def test_is_valid_y_improper_shape_3D_F(self):
        y = np.zeros((3, 3, 3), dtype=np.float64)
        assert_equal(is_valid_y(y), False)

    def test_is_valid_y_correct_2_by_2(self):
        y = self.correct_n_by_n(2)
        assert_equal(is_valid_y(y), True)

    def test_is_valid_y_correct_3_by_3(self):
        y = self.correct_n_by_n(3)
        assert_equal(is_valid_y(y), True)

    def test_is_valid_y_correct_4_by_4(self):
        y = self.correct_n_by_n(4)
        assert_equal(is_valid_y(y), True)

    def test_is_valid_y_correct_5_by_5(self):
        y = self.correct_n_by_n(5)
        assert_equal(is_valid_y(y), True)

    def test_is_valid_y_2_100(self):
        a = set()
        for n in range(2, 16):
            a.add(n * (n - 1) / 2)
        for i in range(5, 105):
            if i not in a:
                with pytest.raises(ValueError):
                    self.bad_y(i)

    def bad_y(self, n):
        y = np.random.rand(n)
        return is_valid_y(y, throw=True)

    def correct_n_by_n(self, n):
        y = np.random.rand(n * (n - 1) // 2)
        return y