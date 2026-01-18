import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_, assert_warns
import pytest
from pytest import raises as assert_raises
import scipy.cluster.hierarchy
from scipy.cluster.hierarchy import (
from scipy.spatial.distance import pdist
from scipy.cluster._hierarchy import Heap
from scipy.conftest import (
from scipy._lib._array_api import xp_assert_close
from . import hierarchy_test_data
class TestIsIsomorphic:

    @skip_if_array_api
    def test_is_isomorphic_1(self):
        a = [1, 1, 1]
        b = [2, 2, 2]
        assert_(is_isomorphic(a, b))
        assert_(is_isomorphic(b, a))

    @skip_if_array_api
    def test_is_isomorphic_2(self):
        a = np.asarray([1, 7, 1])
        b = np.asarray([2, 3, 2])
        assert_(is_isomorphic(a, b))
        assert_(is_isomorphic(b, a))

    @skip_if_array_api
    def test_is_isomorphic_3(self):
        a = np.asarray([])
        b = np.asarray([])
        assert_(is_isomorphic(a, b))

    @skip_if_array_api
    def test_is_isomorphic_4A(self):
        a = np.asarray([1, 2, 3])
        b = np.asarray([1, 3, 2])
        assert_(is_isomorphic(a, b))
        assert_(is_isomorphic(b, a))

    @skip_if_array_api
    def test_is_isomorphic_4B(self):
        a = np.asarray([1, 2, 3, 3])
        b = np.asarray([1, 3, 2, 3])
        assert_(is_isomorphic(a, b) is False)
        assert_(is_isomorphic(b, a) is False)

    @skip_if_array_api
    def test_is_isomorphic_4C(self):
        a = np.asarray([7, 2, 3])
        b = np.asarray([6, 3, 2])
        assert_(is_isomorphic(a, b))
        assert_(is_isomorphic(b, a))

    @skip_if_array_api
    def test_is_isomorphic_5(self):
        for nc in [2, 3, 5]:
            self.help_is_isomorphic_randperm(1000, nc)

    @skip_if_array_api
    def test_is_isomorphic_6(self):
        for nc in [2, 3, 5]:
            self.help_is_isomorphic_randperm(1000, nc, True, 5)

    @skip_if_array_api
    def test_is_isomorphic_7(self):
        a = np.asarray([1, 2, 3])
        b = np.asarray([1, 1, 1])
        assert_(not is_isomorphic(a, b))

    def help_is_isomorphic_randperm(self, nobs, nclusters, noniso=False, nerrors=0):
        for k in range(3):
            a = (np.random.rand(nobs) * nclusters).astype(int)
            b = np.zeros(a.size, dtype=int)
            P = np.random.permutation(nclusters)
            for i in range(0, a.shape[0]):
                b[i] = P[a[i]]
            if noniso:
                Q = np.random.permutation(nobs)
                b[Q[0:nerrors]] += 1
                b[Q[0:nerrors]] %= nclusters
            assert_(is_isomorphic(a, b) == (not noniso))
            assert_(is_isomorphic(b, a) == (not noniso))