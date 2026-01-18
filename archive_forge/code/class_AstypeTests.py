import re
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import (
from pandas import (
import pandas._testing as tm
class AstypeTests:
    """Tests common to IntervalIndex with any subtype"""

    def test_astype_idempotent(self, index):
        result = index.astype('interval')
        tm.assert_index_equal(result, index)
        result = index.astype(index.dtype)
        tm.assert_index_equal(result, index)

    def test_astype_object(self, index):
        result = index.astype(object)
        expected = Index(index.values, dtype='object')
        tm.assert_index_equal(result, expected)
        assert not result.equals(index)

    def test_astype_category(self, index):
        result = index.astype('category')
        expected = CategoricalIndex(index.values)
        tm.assert_index_equal(result, expected)
        result = index.astype(CategoricalDtype())
        tm.assert_index_equal(result, expected)
        categories = index.dropna().unique().values[:-1]
        dtype = CategoricalDtype(categories=categories, ordered=True)
        result = index.astype(dtype)
        expected = CategoricalIndex(index.values, categories=categories, ordered=True)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('dtype', ['int64', 'uint64', 'float64', 'complex128', 'period[M]', 'timedelta64', 'timedelta64[ns]', 'datetime64', 'datetime64[ns]', 'datetime64[ns, US/Eastern]'])
    def test_astype_cannot_cast(self, index, dtype):
        msg = 'Cannot cast IntervalIndex to dtype'
        with pytest.raises(TypeError, match=msg):
            index.astype(dtype)

    def test_astype_invalid_dtype(self, index):
        msg = 'data type ["\']fake_dtype["\'] not understood'
        with pytest.raises(TypeError, match=msg):
            index.astype('fake_dtype')