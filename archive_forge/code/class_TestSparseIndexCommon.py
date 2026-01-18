import operator
import numpy as np
import pytest
import pandas._libs.sparse as splib
import pandas.util._test_decorators as td
from pandas import Series
import pandas._testing as tm
from pandas.core.arrays.sparse import (
class TestSparseIndexCommon:

    def test_int_internal(self):
        idx = make_sparse_index(4, np.array([2, 3], dtype=np.int32), kind='integer')
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 2
        tm.assert_numpy_array_equal(idx.indices, np.array([2, 3], dtype=np.int32))
        idx = make_sparse_index(4, np.array([], dtype=np.int32), kind='integer')
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 0
        tm.assert_numpy_array_equal(idx.indices, np.array([], dtype=np.int32))
        idx = make_sparse_index(4, np.array([0, 1, 2, 3], dtype=np.int32), kind='integer')
        assert isinstance(idx, IntIndex)
        assert idx.npoints == 4
        tm.assert_numpy_array_equal(idx.indices, np.array([0, 1, 2, 3], dtype=np.int32))

    def test_block_internal(self):
        idx = make_sparse_index(4, np.array([2, 3], dtype=np.int32), kind='block')
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 2
        tm.assert_numpy_array_equal(idx.blocs, np.array([2], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([2], dtype=np.int32))
        idx = make_sparse_index(4, np.array([], dtype=np.int32), kind='block')
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 0
        tm.assert_numpy_array_equal(idx.blocs, np.array([], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([], dtype=np.int32))
        idx = make_sparse_index(4, np.array([0, 1, 2, 3], dtype=np.int32), kind='block')
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 4
        tm.assert_numpy_array_equal(idx.blocs, np.array([0], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([4], dtype=np.int32))
        idx = make_sparse_index(4, np.array([0, 2, 3], dtype=np.int32), kind='block')
        assert isinstance(idx, BlockIndex)
        assert idx.npoints == 3
        tm.assert_numpy_array_equal(idx.blocs, np.array([0, 2], dtype=np.int32))
        tm.assert_numpy_array_equal(idx.blengths, np.array([1, 2], dtype=np.int32))

    @pytest.mark.parametrize('kind', ['integer', 'block'])
    def test_lookup(self, kind):
        idx = make_sparse_index(4, np.array([2, 3], dtype=np.int32), kind=kind)
        assert idx.lookup(-1) == -1
        assert idx.lookup(0) == -1
        assert idx.lookup(1) == -1
        assert idx.lookup(2) == 0
        assert idx.lookup(3) == 1
        assert idx.lookup(4) == -1
        idx = make_sparse_index(4, np.array([], dtype=np.int32), kind=kind)
        for i in range(-1, 5):
            assert idx.lookup(i) == -1
        idx = make_sparse_index(4, np.array([0, 1, 2, 3], dtype=np.int32), kind=kind)
        assert idx.lookup(-1) == -1
        assert idx.lookup(0) == 0
        assert idx.lookup(1) == 1
        assert idx.lookup(2) == 2
        assert idx.lookup(3) == 3
        assert idx.lookup(4) == -1
        idx = make_sparse_index(4, np.array([0, 2, 3], dtype=np.int32), kind=kind)
        assert idx.lookup(-1) == -1
        assert idx.lookup(0) == 0
        assert idx.lookup(1) == -1
        assert idx.lookup(2) == 1
        assert idx.lookup(3) == 2
        assert idx.lookup(4) == -1

    @pytest.mark.parametrize('kind', ['integer', 'block'])
    def test_lookup_array(self, kind):
        idx = make_sparse_index(4, np.array([2, 3], dtype=np.int32), kind=kind)
        res = idx.lookup_array(np.array([-1, 0, 2], dtype=np.int32))
        exp = np.array([-1, -1, 0], dtype=np.int32)
        tm.assert_numpy_array_equal(res, exp)
        res = idx.lookup_array(np.array([4, 2, 1, 3], dtype=np.int32))
        exp = np.array([-1, 0, -1, 1], dtype=np.int32)
        tm.assert_numpy_array_equal(res, exp)
        idx = make_sparse_index(4, np.array([], dtype=np.int32), kind=kind)
        res = idx.lookup_array(np.array([-1, 0, 2, 4], dtype=np.int32))
        exp = np.array([-1, -1, -1, -1], dtype=np.int32)
        tm.assert_numpy_array_equal(res, exp)
        idx = make_sparse_index(4, np.array([0, 1, 2, 3], dtype=np.int32), kind=kind)
        res = idx.lookup_array(np.array([-1, 0, 2], dtype=np.int32))
        exp = np.array([-1, 0, 2], dtype=np.int32)
        tm.assert_numpy_array_equal(res, exp)
        res = idx.lookup_array(np.array([4, 2, 1, 3], dtype=np.int32))
        exp = np.array([-1, 2, 1, 3], dtype=np.int32)
        tm.assert_numpy_array_equal(res, exp)
        idx = make_sparse_index(4, np.array([0, 2, 3], dtype=np.int32), kind=kind)
        res = idx.lookup_array(np.array([2, 1, 3, 0], dtype=np.int32))
        exp = np.array([1, -1, 2, 0], dtype=np.int32)
        tm.assert_numpy_array_equal(res, exp)
        res = idx.lookup_array(np.array([1, 4, 2, 5], dtype=np.int32))
        exp = np.array([-1, -1, 1, -1], dtype=np.int32)
        tm.assert_numpy_array_equal(res, exp)

    @pytest.mark.parametrize('idx, expected', [[0, -1], [5, 0], [7, 2], [8, -1], [9, -1], [10, -1], [11, -1], [12, 3], [17, 8], [18, -1]])
    def test_lookup_basics(self, idx, expected):
        bindex = BlockIndex(20, [5, 12], [3, 6])
        assert bindex.lookup(idx) == expected
        iindex = bindex.to_int_index()
        assert iindex.lookup(idx) == expected