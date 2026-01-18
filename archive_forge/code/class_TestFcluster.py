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
class TestFcluster:

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_fclusterdata(self, xp):
        for t in hierarchy_test_data.fcluster_inconsistent:
            self.check_fclusterdata(t, 'inconsistent', xp)
        for t in hierarchy_test_data.fcluster_distance:
            self.check_fclusterdata(t, 'distance', xp)
        for t in hierarchy_test_data.fcluster_maxclust:
            self.check_fclusterdata(t, 'maxclust', xp)

    def check_fclusterdata(self, t, criterion, xp):
        expectedT = xp.asarray(getattr(hierarchy_test_data, 'fcluster_' + criterion)[t])
        X = xp.asarray(hierarchy_test_data.Q_X)
        T = fclusterdata(X, criterion=criterion, t=t)
        assert_(is_isomorphic(T, expectedT))

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_fcluster(self, xp):
        for t in hierarchy_test_data.fcluster_inconsistent:
            self.check_fcluster(t, 'inconsistent', xp)
        for t in hierarchy_test_data.fcluster_distance:
            self.check_fcluster(t, 'distance', xp)
        for t in hierarchy_test_data.fcluster_maxclust:
            self.check_fcluster(t, 'maxclust', xp)

    def check_fcluster(self, t, criterion, xp):
        expectedT = xp.asarray(getattr(hierarchy_test_data, 'fcluster_' + criterion)[t])
        Z = single(xp.asarray(hierarchy_test_data.Q_X))
        T = fcluster(Z, criterion=criterion, t=t)
        assert_(is_isomorphic(T, expectedT))

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_fcluster_monocrit(self, xp):
        for t in hierarchy_test_data.fcluster_distance:
            self.check_fcluster_monocrit(t, xp)
        for t in hierarchy_test_data.fcluster_maxclust:
            self.check_fcluster_maxclust_monocrit(t, xp)

    def check_fcluster_monocrit(self, t, xp):
        expectedT = xp.asarray(hierarchy_test_data.fcluster_distance[t])
        Z = single(xp.asarray(hierarchy_test_data.Q_X))
        T = fcluster(Z, t, criterion='monocrit', monocrit=maxdists(Z))
        assert_(is_isomorphic(T, expectedT))

    def check_fcluster_maxclust_monocrit(self, t, xp):
        expectedT = xp.asarray(hierarchy_test_data.fcluster_maxclust[t])
        Z = single(xp.asarray(hierarchy_test_data.Q_X))
        T = fcluster(Z, t, criterion='maxclust_monocrit', monocrit=maxdists(Z))
        assert_(is_isomorphic(T, expectedT))