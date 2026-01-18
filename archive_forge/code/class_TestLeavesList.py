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
class TestLeavesList:

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_leaves_list_1x4(self, xp):
        Z = xp.asarray([[0, 1, 3.0, 2]], dtype=xp.float64)
        to_tree(Z)
        assert_allclose(leaves_list(Z), [0, 1], rtol=1e-15)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_leaves_list_2x4(self, xp):
        Z = xp.asarray([[0, 1, 3.0, 2], [3, 2, 4.0, 3]], dtype=xp.float64)
        to_tree(Z)
        assert_allclose(leaves_list(Z), [0, 1, 2], rtol=1e-15)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_leaves_list_Q(self, xp):
        for method in ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']:
            self.check_leaves_list_Q(method, xp)

    def check_leaves_list_Q(self, method, xp):
        X = xp.asarray(hierarchy_test_data.Q_X)
        Z = linkage(X, method)
        node = to_tree(Z)
        assert_allclose(node.pre_order(), leaves_list(Z), rtol=1e-15)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_Q_subtree_pre_order(self, xp):
        X = xp.asarray(hierarchy_test_data.Q_X)
        Z = linkage(X, 'single')
        node = to_tree(Z)
        assert_allclose(node.pre_order(), node.get_left().pre_order() + node.get_right().pre_order(), rtol=1e-15)