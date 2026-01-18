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
class TestIsValidInconsistent:

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_is_valid_im_int_type(self, xp):
        R = xp.asarray([[0, 1, 3.0, 2], [3, 2, 4.0, 3]], dtype=xp.int64)
        assert_(is_valid_im(R) is False)
        assert_raises(TypeError, is_valid_im, R, throw=True)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_is_valid_im_various_size(self, xp):
        for nrow, ncol, valid in [(2, 5, False), (2, 3, False), (1, 4, True), (2, 4, True)]:
            self.check_is_valid_im_various_size(nrow, ncol, valid, xp)

    def check_is_valid_im_various_size(self, nrow, ncol, valid, xp):
        R = xp.asarray([[0, 1, 3.0, 2, 5], [3, 2, 4.0, 3, 3]], dtype=xp.float64)
        R = R[:nrow, :ncol]
        assert_(is_valid_im(R) == valid)
        if not valid:
            assert_raises(ValueError, is_valid_im, R, throw=True)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_is_valid_im_empty(self, xp):
        R = xp.zeros((0, 4), dtype=xp.float64)
        assert_(is_valid_im(R) is False)
        assert_raises(ValueError, is_valid_im, R, throw=True)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_is_valid_im_4_and_up(self, xp):
        for i in range(4, 15, 3):
            y = np.random.rand(i * (i - 1) // 2)
            y = xp.asarray(y)
            Z = linkage(y)
            R = inconsistent(Z)
            assert_(is_valid_im(R) is True)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_is_valid_im_4_and_up_neg_index_left(self, xp):
        for i in range(4, 15, 3):
            y = np.random.rand(i * (i - 1) // 2)
            y = xp.asarray(y)
            Z = linkage(y)
            R = inconsistent(Z)
            R[i // 2, 0] = -2.0
            assert_(is_valid_im(R) is False)
            assert_raises(ValueError, is_valid_im, R, throw=True)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_is_valid_im_4_and_up_neg_index_right(self, xp):
        for i in range(4, 15, 3):
            y = np.random.rand(i * (i - 1) // 2)
            y = xp.asarray(y)
            Z = linkage(y)
            R = inconsistent(Z)
            R[i // 2, 1] = -2.0
            assert_(is_valid_im(R) is False)
            assert_raises(ValueError, is_valid_im, R, throw=True)

    @skip_if_array_api_gpu
    @array_api_compatible
    def test_is_valid_im_4_and_up_neg_dist(self, xp):
        for i in range(4, 15, 3):
            y = np.random.rand(i * (i - 1) // 2)
            y = xp.asarray(y)
            Z = linkage(y)
            R = inconsistent(Z)
            R[i // 2, 2] = -0.5
            assert_(is_valid_im(R) is False)
            assert_raises(ValueError, is_valid_im, R, throw=True)