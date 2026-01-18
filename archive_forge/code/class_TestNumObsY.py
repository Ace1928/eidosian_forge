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
class TestNumObsY:

    def test_num_obs_y_multi_matrix(self):
        for n in range(2, 10):
            X = np.random.rand(n, 4)
            Y = wpdist_no_const(X)
            assert_equal(num_obs_y(Y), n)

    def test_num_obs_y_1(self):
        with pytest.raises(ValueError):
            self.check_y(1)

    def test_num_obs_y_2(self):
        assert_(self.check_y(2))

    def test_num_obs_y_3(self):
        assert_(self.check_y(3))

    def test_num_obs_y_4(self):
        assert_(self.check_y(4))

    def test_num_obs_y_5_10(self):
        for i in range(5, 16):
            self.minit(i)

    def test_num_obs_y_2_100(self):
        a = set()
        for n in range(2, 16):
            a.add(n * (n - 1) / 2)
        for i in range(5, 105):
            if i not in a:
                with pytest.raises(ValueError):
                    self.bad_y(i)

    def minit(self, n):
        assert_(self.check_y(n))

    def bad_y(self, n):
        y = np.random.rand(n)
        return num_obs_y(y)

    def check_y(self, n):
        return num_obs_y(self.make_y(n)) == n

    def make_y(self, n):
        return np.random.rand(n * (n - 1) // 2)