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
class TestNumObsDM:

    def test_num_obs_dm_multi_matrix(self):
        for n in range(1, 10):
            X = np.random.rand(n, 4)
            Y = wpdist_no_const(X)
            A = squareform(Y)
            if verbose >= 3:
                print(A.shape, Y.shape)
            assert_equal(num_obs_dm(A), n)

    def test_num_obs_dm_0(self):
        assert_(self.check_D(0))

    def test_num_obs_dm_1(self):
        assert_(self.check_D(1))

    def test_num_obs_dm_2(self):
        assert_(self.check_D(2))

    def test_num_obs_dm_3(self):
        assert_(self.check_D(2))

    def test_num_obs_dm_4(self):
        assert_(self.check_D(4))

    def check_D(self, n):
        return num_obs_dm(self.make_D(n)) == n

    def make_D(self, n):
        return np.random.rand(n, n)