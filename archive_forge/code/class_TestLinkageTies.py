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
class TestLinkageTies:
    _expectations = {'single': np.array([[0, 1, 1.41421356, 2], [2, 3, 1.41421356, 3]]), 'complete': np.array([[0, 1, 1.41421356, 2], [2, 3, 2.82842712, 3]]), 'average': np.array([[0, 1, 1.41421356, 2], [2, 3, 2.12132034, 3]]), 'weighted': np.array([[0, 1, 1.41421356, 2], [2, 3, 2.12132034, 3]]), 'centroid': np.array([[0, 1, 1.41421356, 2], [2, 3, 2.12132034, 3]]), 'median': np.array([[0, 1, 1.41421356, 2], [2, 3, 2.12132034, 3]]), 'ward': np.array([[0, 1, 1.41421356, 2], [2, 3, 2.44948974, 3]])}

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_linkage_ties(self, xp):
        for method in ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']:
            self.check_linkage_ties(method, xp)

    def check_linkage_ties(self, method, xp):
        X = xp.asarray([[-1, -1], [0, 0], [1, 1]])
        Z = linkage(X, method=method)
        expectedZ = self._expectations[method]
        xp_assert_close(Z, xp.asarray(expectedZ), atol=1e-06)