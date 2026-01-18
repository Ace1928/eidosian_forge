import re
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import (
from pandas import (
import pandas._testing as tm
class TestIntSubtype(AstypeTests):
    """Tests specific to IntervalIndex with integer-like subtype"""
    indexes = [IntervalIndex.from_breaks(np.arange(-10, 11, dtype='int64')), IntervalIndex.from_breaks(np.arange(100, dtype='uint64'), closed='left')]

    @pytest.fixture(params=indexes)
    def index(self, request):
        return request.param

    @pytest.mark.parametrize('subtype', ['float64', 'datetime64[ns]', 'timedelta64[ns]'])
    def test_subtype_conversion(self, index, subtype):
        dtype = IntervalDtype(subtype, index.closed)
        result = index.astype(dtype)
        expected = IntervalIndex.from_arrays(index.left.astype(subtype), index.right.astype(subtype), closed=index.closed)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('subtype_start, subtype_end', [('int64', 'uint64'), ('uint64', 'int64')])
    def test_subtype_integer(self, subtype_start, subtype_end):
        index = IntervalIndex.from_breaks(np.arange(100, dtype=subtype_start))
        dtype = IntervalDtype(subtype_end, index.closed)
        result = index.astype(dtype)
        expected = IntervalIndex.from_arrays(index.left.astype(subtype_end), index.right.astype(subtype_end), closed=index.closed)
        tm.assert_index_equal(result, expected)

    @pytest.mark.xfail(reason='GH#15832')
    def test_subtype_integer_errors(self):
        index = interval_range(-10, 10)
        dtype = IntervalDtype('uint64', 'right')
        msg = '^(?!(left side of interval must be <= right side))'
        with pytest.raises(ValueError, match=msg):
            index.astype(dtype)