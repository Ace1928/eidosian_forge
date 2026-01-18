import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
class TestSeriesToXArray:

    def test_to_xarray_index_types(self, index_flat):
        index = index_flat
        from xarray import DataArray
        ser = Series(range(len(index)), index=index, dtype='int64')
        ser.index.name = 'foo'
        result = ser.to_xarray()
        repr(result)
        assert len(result) == len(index)
        assert len(result.coords) == 1
        tm.assert_almost_equal(list(result.coords.keys()), ['foo'])
        assert isinstance(result, DataArray)
        tm.assert_series_equal(result.to_series(), ser)

    def test_to_xarray_empty(self):
        from xarray import DataArray
        ser = Series([], dtype=object)
        ser.index.name = 'foo'
        result = ser.to_xarray()
        assert len(result) == 0
        assert len(result.coords) == 1
        tm.assert_almost_equal(list(result.coords.keys()), ['foo'])
        assert isinstance(result, DataArray)

    def test_to_xarray_with_multiindex(self):
        from xarray import DataArray
        mi = MultiIndex.from_product([['a', 'b'], range(3)], names=['one', 'two'])
        ser = Series(range(6), dtype='int64', index=mi)
        result = ser.to_xarray()
        assert len(result) == 2
        tm.assert_almost_equal(list(result.coords.keys()), ['one', 'two'])
        assert isinstance(result, DataArray)
        res = result.to_series()
        tm.assert_series_equal(res, ser)