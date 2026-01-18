import numpy as np
import pytest
from pandas._libs import join as libjoin
from pandas._libs.join import (
import pandas._testing as tm
class TestIndexer:

    @pytest.mark.parametrize('dtype', ['int32', 'int64', 'float32', 'float64', 'object'])
    def test_outer_join_indexer(self, dtype):
        indexer = libjoin.outer_join_indexer
        left = np.arange(3, dtype=dtype)
        right = np.arange(2, 5, dtype=dtype)
        empty = np.array([], dtype=dtype)
        result, lindexer, rindexer = indexer(left, right)
        assert isinstance(result, np.ndarray)
        assert isinstance(lindexer, np.ndarray)
        assert isinstance(rindexer, np.ndarray)
        tm.assert_numpy_array_equal(result, np.arange(5, dtype=dtype))
        exp = np.array([0, 1, 2, -1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(lindexer, exp)
        exp = np.array([-1, -1, 0, 1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(rindexer, exp)
        result, lindexer, rindexer = indexer(empty, right)
        tm.assert_numpy_array_equal(result, right)
        exp = np.array([-1, -1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(lindexer, exp)
        exp = np.array([0, 1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(rindexer, exp)
        result, lindexer, rindexer = indexer(left, empty)
        tm.assert_numpy_array_equal(result, left)
        exp = np.array([0, 1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(lindexer, exp)
        exp = np.array([-1, -1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(rindexer, exp)

    def test_cython_left_outer_join(self):
        left = np.array([0, 1, 2, 1, 2, 0, 0, 1, 2, 3, 3], dtype=np.intp)
        right = np.array([1, 1, 0, 4, 2, 2, 1], dtype=np.intp)
        max_group = 5
        ls, rs = left_outer_join(left, right, max_group)
        exp_ls = left.argsort(kind='mergesort')
        exp_rs = right.argsort(kind='mergesort')
        exp_li = np.array([0, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8, 9, 10])
        exp_ri = np.array([0, 0, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 4, 5, 4, 5, -1, -1])
        exp_ls = exp_ls.take(exp_li)
        exp_ls[exp_li == -1] = -1
        exp_rs = exp_rs.take(exp_ri)
        exp_rs[exp_ri == -1] = -1
        tm.assert_numpy_array_equal(ls, exp_ls, check_dtype=False)
        tm.assert_numpy_array_equal(rs, exp_rs, check_dtype=False)

    def test_cython_right_outer_join(self):
        left = np.array([0, 1, 2, 1, 2, 0, 0, 1, 2, 3, 3], dtype=np.intp)
        right = np.array([1, 1, 0, 4, 2, 2, 1], dtype=np.intp)
        max_group = 5
        rs, ls = left_outer_join(right, left, max_group)
        exp_ls = left.argsort(kind='mergesort')
        exp_rs = right.argsort(kind='mergesort')
        exp_li = np.array([0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5, 6, 7, 8, 6, 7, 8, -1])
        exp_ri = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6])
        exp_ls = exp_ls.take(exp_li)
        exp_ls[exp_li == -1] = -1
        exp_rs = exp_rs.take(exp_ri)
        exp_rs[exp_ri == -1] = -1
        tm.assert_numpy_array_equal(ls, exp_ls)
        tm.assert_numpy_array_equal(rs, exp_rs)

    def test_cython_inner_join(self):
        left = np.array([0, 1, 2, 1, 2, 0, 0, 1, 2, 3, 3], dtype=np.intp)
        right = np.array([1, 1, 0, 4, 2, 2, 1, 4], dtype=np.intp)
        max_group = 5
        ls, rs = inner_join(left, right, max_group)
        exp_ls = left.argsort(kind='mergesort')
        exp_rs = right.argsort(kind='mergesort')
        exp_li = np.array([0, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8])
        exp_ri = np.array([0, 0, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 4, 5, 4, 5])
        exp_ls = exp_ls.take(exp_li)
        exp_ls[exp_li == -1] = -1
        exp_rs = exp_rs.take(exp_ri)
        exp_rs[exp_ri == -1] = -1
        tm.assert_numpy_array_equal(ls, exp_ls)
        tm.assert_numpy_array_equal(rs, exp_rs)