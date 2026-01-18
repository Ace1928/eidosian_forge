import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
class TestBroadcastedAssignments:

    def assign(self, a, ind, val):
        a[ind] = val
        return a

    def test_prepending_ones(self):
        a = np.zeros((3, 2))
        a[...] = np.ones((1, 3, 2))
        a[[0, 1, 2], :] = np.ones((1, 3, 2))
        a[:, [0, 1]] = np.ones((1, 3, 2))
        a[[[0], [1], [2]], [0, 1]] = np.ones((1, 3, 2))

    def test_prepend_not_one(self):
        assign = self.assign
        s_ = np.s_
        a = np.zeros(5)
        assert_raises(ValueError, assign, a, s_[...], np.ones((2, 1)))
        assert_raises(ValueError, assign, a, s_[[1, 2, 3],], np.ones((2, 1)))
        assert_raises(ValueError, assign, a, s_[[[1], [2]],], np.ones((2, 2, 1)))

    def test_simple_broadcasting_errors(self):
        assign = self.assign
        s_ = np.s_
        a = np.zeros((5, 1))
        assert_raises(ValueError, assign, a, s_[...], np.zeros((5, 2)))
        assert_raises(ValueError, assign, a, s_[...], np.zeros((5, 0)))
        assert_raises(ValueError, assign, a, s_[:, [0]], np.zeros((5, 2)))
        assert_raises(ValueError, assign, a, s_[:, [0]], np.zeros((5, 0)))
        assert_raises(ValueError, assign, a, s_[[0], :], np.zeros((2, 1)))

    @pytest.mark.parametrize('index', [(..., [1, 2], slice(None)), ([0, 1], ..., 0), (..., [1, 2], [1, 2])])
    def test_broadcast_error_reports_correct_shape(self, index):
        values = np.zeros((100, 100))
        arr = np.zeros((3, 4, 5, 6, 7))
        shape_str = str(arr[index].shape).replace(' ', '')
        with pytest.raises(ValueError) as e:
            arr[index] = values
        assert str(e.value).endswith(shape_str)

    def test_index_is_larger(self):
        a = np.zeros((5, 5))
        a[[[0], [1], [2]], [0, 1, 2]] = [2, 3, 4]
        assert_((a[:3, :3] == [2, 3, 4]).all())

    def test_broadcast_subspace(self):
        a = np.zeros((100, 100))
        v = np.arange(100)[:, None]
        b = np.arange(100)[::-1]
        a[b] = v
        assert_((a[::-1] == v).all())