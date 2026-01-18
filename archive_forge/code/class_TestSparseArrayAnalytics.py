import re
import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
class TestSparseArrayAnalytics:

    @pytest.mark.parametrize('data,expected', [(np.array([1, 2, 3, 4, 5], dtype=float), SparseArray(np.array([1.0, 3.0, 6.0, 10.0, 15.0]))), (np.array([1, 2, np.nan, 4, 5], dtype=float), SparseArray(np.array([1.0, 3.0, np.nan, 7.0, 12.0])))])
    @pytest.mark.parametrize('numpy', [True, False])
    def test_cumsum(self, data, expected, numpy):
        cumsum = np.cumsum if numpy else lambda s: s.cumsum()
        out = cumsum(SparseArray(data))
        tm.assert_sp_array_equal(out, expected)
        out = cumsum(SparseArray(data, fill_value=np.nan))
        tm.assert_sp_array_equal(out, expected)
        out = cumsum(SparseArray(data, fill_value=2))
        tm.assert_sp_array_equal(out, expected)
        if numpy:
            msg = "the 'dtype' parameter is not supported"
            with pytest.raises(ValueError, match=msg):
                np.cumsum(SparseArray(data), dtype=np.int64)
            msg = "the 'out' parameter is not supported"
            with pytest.raises(ValueError, match=msg):
                np.cumsum(SparseArray(data), out=out)
        else:
            axis = 1
            msg = re.escape(f'axis(={axis}) out of bounds')
            with pytest.raises(ValueError, match=msg):
                SparseArray(data).cumsum(axis=axis)

    def test_ufunc(self):
        sparse = SparseArray([1, np.nan, 2, np.nan, -2])
        result = SparseArray([1, np.nan, 2, np.nan, 2])
        tm.assert_sp_array_equal(abs(sparse), result)
        tm.assert_sp_array_equal(np.abs(sparse), result)
        sparse = SparseArray([1, -1, 2, -2], fill_value=1)
        result = SparseArray([1, 2, 2], sparse_index=sparse.sp_index, fill_value=1)
        tm.assert_sp_array_equal(abs(sparse), result)
        tm.assert_sp_array_equal(np.abs(sparse), result)
        sparse = SparseArray([1, -1, 2, -2], fill_value=-1)
        exp = SparseArray([1, 1, 2, 2], fill_value=1)
        tm.assert_sp_array_equal(abs(sparse), exp)
        tm.assert_sp_array_equal(np.abs(sparse), exp)
        sparse = SparseArray([1, np.nan, 2, np.nan, -2])
        result = SparseArray(np.sin([1, np.nan, 2, np.nan, -2]))
        tm.assert_sp_array_equal(np.sin(sparse), result)
        sparse = SparseArray([1, -1, 2, -2], fill_value=1)
        result = SparseArray(np.sin([1, -1, 2, -2]), fill_value=np.sin(1))
        tm.assert_sp_array_equal(np.sin(sparse), result)
        sparse = SparseArray([1, -1, 0, -2], fill_value=0)
        result = SparseArray(np.sin([1, -1, 0, -2]), fill_value=np.sin(0))
        tm.assert_sp_array_equal(np.sin(sparse), result)

    def test_ufunc_args(self):
        sparse = SparseArray([1, np.nan, 2, np.nan, -2])
        result = SparseArray([2, np.nan, 3, np.nan, -1])
        tm.assert_sp_array_equal(np.add(sparse, 1), result)
        sparse = SparseArray([1, -1, 2, -2], fill_value=1)
        result = SparseArray([2, 0, 3, -1], fill_value=2)
        tm.assert_sp_array_equal(np.add(sparse, 1), result)
        sparse = SparseArray([1, -1, 0, -2], fill_value=0)
        result = SparseArray([2, 0, 1, -1], fill_value=1)
        tm.assert_sp_array_equal(np.add(sparse, 1), result)

    @pytest.mark.parametrize('fill_value', [0.0, np.nan])
    def test_modf(self, fill_value):
        sparse = SparseArray([fill_value] * 10 + [1.1, 2.2], fill_value=fill_value)
        r1, r2 = np.modf(sparse)
        e1, e2 = np.modf(np.asarray(sparse))
        tm.assert_sp_array_equal(r1, SparseArray(e1, fill_value=fill_value))
        tm.assert_sp_array_equal(r2, SparseArray(e2, fill_value=fill_value))

    def test_nbytes_integer(self):
        arr = SparseArray([1, 0, 0, 0, 2], kind='integer')
        result = arr.nbytes
        assert result == 24

    def test_nbytes_block(self):
        arr = SparseArray([1, 2, 0, 0, 0], kind='block')
        result = arr.nbytes
        assert result == 24

    def test_asarray_datetime64(self):
        s = SparseArray(pd.to_datetime(['2012', None, None, '2013']))
        np.asarray(s)

    def test_density(self):
        arr = SparseArray([0, 1])
        assert arr.density == 0.5

    def test_npoints(self):
        arr = SparseArray([0, 1])
        assert arr.npoints == 1