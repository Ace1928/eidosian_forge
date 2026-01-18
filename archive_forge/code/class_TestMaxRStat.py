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
class TestMaxRStat:

    @array_api_compatible
    def test_maxRstat_invalid_index(self, xp):
        for i in [3.3, -1, 4]:
            self.check_maxRstat_invalid_index(i, xp)

    def check_maxRstat_invalid_index(self, i, xp):
        Z = xp.asarray([[0, 1, 0.3, 4]], dtype=xp.float64)
        R = xp.asarray([[0, 0, 0, 0.3]], dtype=xp.float64)
        if isinstance(i, int):
            assert_raises(ValueError, maxRstat, Z, R, i)
        else:
            assert_raises(TypeError, maxRstat, Z, R, i)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_maxRstat_empty_linkage(self, xp):
        for i in range(4):
            self.check_maxRstat_empty_linkage(i, xp)

    def check_maxRstat_empty_linkage(self, i, xp):
        Z = xp.zeros((0, 4), dtype=xp.float64)
        R = xp.zeros((0, 4), dtype=xp.float64)
        assert_raises(ValueError, maxRstat, Z, R, i)

    @array_api_compatible
    def test_maxRstat_difrow_linkage(self, xp):
        for i in range(4):
            self.check_maxRstat_difrow_linkage(i, xp)

    def check_maxRstat_difrow_linkage(self, i, xp):
        Z = xp.asarray([[0, 1, 0.3, 4]], dtype=xp.float64)
        R = np.random.rand(2, 4)
        R = xp.asarray(R)
        assert_raises(ValueError, maxRstat, Z, R, i)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_maxRstat_one_cluster_linkage(self, xp):
        for i in range(4):
            self.check_maxRstat_one_cluster_linkage(i, xp)

    def check_maxRstat_one_cluster_linkage(self, i, xp):
        Z = xp.asarray([[0, 1, 0.3, 4]], dtype=xp.float64)
        R = xp.asarray([[0, 0, 0, 0.3]], dtype=xp.float64)
        MD = maxRstat(Z, R, 1)
        expectedMD = calculate_maximum_inconsistencies(Z, R, 1, xp)
        xp_assert_close(MD, expectedMD, atol=1e-15)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_maxRstat_Q_linkage(self, xp):
        for method in ['single', 'complete', 'ward', 'centroid', 'median']:
            for i in range(4):
                self.check_maxRstat_Q_linkage(method, i, xp)

    def check_maxRstat_Q_linkage(self, method, i, xp):
        X = xp.asarray(hierarchy_test_data.Q_X)
        Z = linkage(X, method)
        R = inconsistent(Z)
        MD = maxRstat(Z, R, 1)
        expectedMD = calculate_maximum_inconsistencies(Z, R, 1, xp)
        xp_assert_close(MD, expectedMD, atol=1e-15)