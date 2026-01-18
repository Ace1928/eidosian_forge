import numpy as np
import pytest
import pandas._testing as tm
from pandas.core.indexes.api import Index
class TestJoinInt64Index:

    def test_join_non_unique(self):
        left = Index([4, 4, 3, 3])
        joined, lidx, ridx = left.join(left, return_indexers=True)
        exp_joined = Index([4, 4, 4, 4, 3, 3, 3, 3])
        tm.assert_index_equal(joined, exp_joined)
        exp_lidx = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.intp)
        tm.assert_numpy_array_equal(lidx, exp_lidx)
        exp_ridx = np.array([0, 1, 0, 1, 2, 3, 2, 3], dtype=np.intp)
        tm.assert_numpy_array_equal(ridx, exp_ridx)

    def test_join_inner(self):
        index = Index(range(0, 20, 2), dtype=np.int64)
        other = Index([7, 12, 25, 1, 2, 5], dtype=np.int64)
        other_mono = Index([1, 2, 5, 7, 12, 25], dtype=np.int64)
        res, lidx, ridx = index.join(other, how='inner', return_indexers=True)
        ind = res.argsort()
        res = res.take(ind)
        lidx = lidx.take(ind)
        ridx = ridx.take(ind)
        eres = Index([2, 12], dtype=np.int64)
        elidx = np.array([1, 6], dtype=np.intp)
        eridx = np.array([4, 1], dtype=np.intp)
        assert isinstance(res, Index) and res.dtype == np.int64
        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        tm.assert_numpy_array_equal(ridx, eridx)
        res, lidx, ridx = index.join(other_mono, how='inner', return_indexers=True)
        res2 = index.intersection(other_mono)
        tm.assert_index_equal(res, res2)
        elidx = np.array([1, 6], dtype=np.intp)
        eridx = np.array([1, 4], dtype=np.intp)
        assert isinstance(res, Index) and res.dtype == np.int64
        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        tm.assert_numpy_array_equal(ridx, eridx)

    def test_join_left(self):
        index = Index(range(0, 20, 2), dtype=np.int64)
        other = Index([7, 12, 25, 1, 2, 5], dtype=np.int64)
        other_mono = Index([1, 2, 5, 7, 12, 25], dtype=np.int64)
        res, lidx, ridx = index.join(other, how='left', return_indexers=True)
        eres = index
        eridx = np.array([-1, 4, -1, -1, -1, -1, 1, -1, -1, -1], dtype=np.intp)
        assert isinstance(res, Index) and res.dtype == np.int64
        tm.assert_index_equal(res, eres)
        assert lidx is None
        tm.assert_numpy_array_equal(ridx, eridx)
        res, lidx, ridx = index.join(other_mono, how='left', return_indexers=True)
        eridx = np.array([-1, 1, -1, -1, -1, -1, 4, -1, -1, -1], dtype=np.intp)
        assert isinstance(res, Index) and res.dtype == np.int64
        tm.assert_index_equal(res, eres)
        assert lidx is None
        tm.assert_numpy_array_equal(ridx, eridx)
        idx = Index([1, 1, 2, 5])
        idx2 = Index([1, 2, 5, 7, 9])
        res, lidx, ridx = idx2.join(idx, how='left', return_indexers=True)
        eres = Index([1, 1, 2, 5, 7, 9])
        eridx = np.array([0, 1, 2, 3, -1, -1], dtype=np.intp)
        elidx = np.array([0, 0, 1, 2, 3, 4], dtype=np.intp)
        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        tm.assert_numpy_array_equal(ridx, eridx)

    def test_join_right(self):
        index = Index(range(0, 20, 2), dtype=np.int64)
        other = Index([7, 12, 25, 1, 2, 5], dtype=np.int64)
        other_mono = Index([1, 2, 5, 7, 12, 25], dtype=np.int64)
        res, lidx, ridx = index.join(other, how='right', return_indexers=True)
        eres = other
        elidx = np.array([-1, 6, -1, -1, 1, -1], dtype=np.intp)
        assert isinstance(other, Index) and other.dtype == np.int64
        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        assert ridx is None
        res, lidx, ridx = index.join(other_mono, how='right', return_indexers=True)
        eres = other_mono
        elidx = np.array([-1, 1, -1, -1, 6, -1], dtype=np.intp)
        assert isinstance(other, Index) and other.dtype == np.int64
        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        assert ridx is None
        idx = Index([1, 1, 2, 5])
        idx2 = Index([1, 2, 5, 7, 9])
        res, lidx, ridx = idx.join(idx2, how='right', return_indexers=True)
        eres = Index([1, 1, 2, 5, 7, 9])
        elidx = np.array([0, 1, 2, 3, -1, -1], dtype=np.intp)
        eridx = np.array([0, 0, 1, 2, 3, 4], dtype=np.intp)
        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        tm.assert_numpy_array_equal(ridx, eridx)

    def test_join_non_int_index(self):
        index = Index(range(0, 20, 2), dtype=np.int64)
        other = Index([3, 6, 7, 8, 10], dtype=object)
        outer = index.join(other, how='outer')
        outer2 = other.join(index, how='outer')
        expected = Index([0, 2, 3, 4, 6, 7, 8, 10, 12, 14, 16, 18])
        tm.assert_index_equal(outer, outer2)
        tm.assert_index_equal(outer, expected)
        inner = index.join(other, how='inner')
        inner2 = other.join(index, how='inner')
        expected = Index([6, 8, 10])
        tm.assert_index_equal(inner, inner2)
        tm.assert_index_equal(inner, expected)
        left = index.join(other, how='left')
        tm.assert_index_equal(left, index.astype(object))
        left2 = other.join(index, how='left')
        tm.assert_index_equal(left2, other)
        right = index.join(other, how='right')
        tm.assert_index_equal(right, other)
        right2 = other.join(index, how='right')
        tm.assert_index_equal(right2, index.astype(object))

    def test_join_outer(self):
        index = Index(range(0, 20, 2), dtype=np.int64)
        other = Index([7, 12, 25, 1, 2, 5], dtype=np.int64)
        other_mono = Index([1, 2, 5, 7, 12, 25], dtype=np.int64)
        res, lidx, ridx = index.join(other, how='outer', return_indexers=True)
        noidx_res = index.join(other, how='outer')
        tm.assert_index_equal(res, noidx_res)
        eres = Index([0, 1, 2, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 25], dtype=np.int64)
        elidx = np.array([0, -1, 1, 2, -1, 3, -1, 4, 5, 6, 7, 8, 9, -1], dtype=np.intp)
        eridx = np.array([-1, 3, 4, -1, 5, -1, 0, -1, -1, 1, -1, -1, -1, 2], dtype=np.intp)
        assert isinstance(res, Index) and res.dtype == np.int64
        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        tm.assert_numpy_array_equal(ridx, eridx)
        res, lidx, ridx = index.join(other_mono, how='outer', return_indexers=True)
        noidx_res = index.join(other_mono, how='outer')
        tm.assert_index_equal(res, noidx_res)
        elidx = np.array([0, -1, 1, 2, -1, 3, -1, 4, 5, 6, 7, 8, 9, -1], dtype=np.intp)
        eridx = np.array([-1, 0, 1, -1, 2, -1, 3, -1, -1, 4, -1, -1, -1, 5], dtype=np.intp)
        assert isinstance(res, Index) and res.dtype == np.int64
        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        tm.assert_numpy_array_equal(ridx, eridx)