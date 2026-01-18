import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
class TestArraySplit:

    def test_integer_0_split(self):
        a = np.arange(10)
        assert_raises(ValueError, array_split, a, 0)

    def test_integer_split(self):
        a = np.arange(10)
        res = array_split(a, 1)
        desired = [np.arange(10)]
        compare_results(res, desired)
        res = array_split(a, 2)
        desired = [np.arange(5), np.arange(5, 10)]
        compare_results(res, desired)
        res = array_split(a, 3)
        desired = [np.arange(4), np.arange(4, 7), np.arange(7, 10)]
        compare_results(res, desired)
        res = array_split(a, 4)
        desired = [np.arange(3), np.arange(3, 6), np.arange(6, 8), np.arange(8, 10)]
        compare_results(res, desired)
        res = array_split(a, 5)
        desired = [np.arange(2), np.arange(2, 4), np.arange(4, 6), np.arange(6, 8), np.arange(8, 10)]
        compare_results(res, desired)
        res = array_split(a, 6)
        desired = [np.arange(2), np.arange(2, 4), np.arange(4, 6), np.arange(6, 8), np.arange(8, 9), np.arange(9, 10)]
        compare_results(res, desired)
        res = array_split(a, 7)
        desired = [np.arange(2), np.arange(2, 4), np.arange(4, 6), np.arange(6, 7), np.arange(7, 8), np.arange(8, 9), np.arange(9, 10)]
        compare_results(res, desired)
        res = array_split(a, 8)
        desired = [np.arange(2), np.arange(2, 4), np.arange(4, 5), np.arange(5, 6), np.arange(6, 7), np.arange(7, 8), np.arange(8, 9), np.arange(9, 10)]
        compare_results(res, desired)
        res = array_split(a, 9)
        desired = [np.arange(2), np.arange(2, 3), np.arange(3, 4), np.arange(4, 5), np.arange(5, 6), np.arange(6, 7), np.arange(7, 8), np.arange(8, 9), np.arange(9, 10)]
        compare_results(res, desired)
        res = array_split(a, 10)
        desired = [np.arange(1), np.arange(1, 2), np.arange(2, 3), np.arange(3, 4), np.arange(4, 5), np.arange(5, 6), np.arange(6, 7), np.arange(7, 8), np.arange(8, 9), np.arange(9, 10)]
        compare_results(res, desired)
        res = array_split(a, 11)
        desired = [np.arange(1), np.arange(1, 2), np.arange(2, 3), np.arange(3, 4), np.arange(4, 5), np.arange(5, 6), np.arange(6, 7), np.arange(7, 8), np.arange(8, 9), np.arange(9, 10), np.array([])]
        compare_results(res, desired)

    def test_integer_split_2D_rows(self):
        a = np.array([np.arange(10), np.arange(10)])
        res = array_split(a, 3, axis=0)
        tgt = [np.array([np.arange(10)]), np.array([np.arange(10)]), np.zeros((0, 10))]
        compare_results(res, tgt)
        assert_(a.dtype.type is res[-1].dtype.type)
        res = array_split(a, [0, 1], axis=0)
        tgt = [np.zeros((0, 10)), np.array([np.arange(10)]), np.array([np.arange(10)])]
        compare_results(res, tgt)
        assert_(a.dtype.type is res[-1].dtype.type)

    def test_integer_split_2D_cols(self):
        a = np.array([np.arange(10), np.arange(10)])
        res = array_split(a, 3, axis=-1)
        desired = [np.array([np.arange(4), np.arange(4)]), np.array([np.arange(4, 7), np.arange(4, 7)]), np.array([np.arange(7, 10), np.arange(7, 10)])]
        compare_results(res, desired)

    def test_integer_split_2D_default(self):
        """ This will fail if we change default axis
        """
        a = np.array([np.arange(10), np.arange(10)])
        res = array_split(a, 3)
        tgt = [np.array([np.arange(10)]), np.array([np.arange(10)]), np.zeros((0, 10))]
        compare_results(res, tgt)
        assert_(a.dtype.type is res[-1].dtype.type)

    @pytest.mark.skipif(not IS_64BIT, reason='Needs 64bit platform')
    def test_integer_split_2D_rows_greater_max_int32(self):
        a = np.broadcast_to([0], (1 << 32, 2))
        res = array_split(a, 4)
        chunk = np.broadcast_to([0], (1 << 30, 2))
        tgt = [chunk] * 4
        for i in range(len(tgt)):
            assert_equal(res[i].shape, tgt[i].shape)

    def test_index_split_simple(self):
        a = np.arange(10)
        indices = [1, 5, 7]
        res = array_split(a, indices, axis=-1)
        desired = [np.arange(0, 1), np.arange(1, 5), np.arange(5, 7), np.arange(7, 10)]
        compare_results(res, desired)

    def test_index_split_low_bound(self):
        a = np.arange(10)
        indices = [0, 5, 7]
        res = array_split(a, indices, axis=-1)
        desired = [np.array([]), np.arange(0, 5), np.arange(5, 7), np.arange(7, 10)]
        compare_results(res, desired)

    def test_index_split_high_bound(self):
        a = np.arange(10)
        indices = [0, 5, 7, 10, 12]
        res = array_split(a, indices, axis=-1)
        desired = [np.array([]), np.arange(0, 5), np.arange(5, 7), np.arange(7, 10), np.array([]), np.array([])]
        compare_results(res, desired)