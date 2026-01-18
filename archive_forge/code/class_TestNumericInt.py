import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestNumericInt:

    @pytest.fixture(params=[np.int64, np.int32, np.int16, np.int8, np.uint64])
    def dtype(self, request):
        return request.param

    @pytest.fixture
    def simple_index(self, dtype):
        return Index(range(0, 20, 2), dtype=dtype)

    def test_is_monotonic(self):
        index_cls = Index
        index = index_cls([1, 2, 3, 4])
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_increasing is True
        assert index._is_strictly_monotonic_increasing is True
        assert index.is_monotonic_decreasing is False
        assert index._is_strictly_monotonic_decreasing is False
        index = index_cls([4, 3, 2, 1])
        assert index.is_monotonic_increasing is False
        assert index._is_strictly_monotonic_increasing is False
        assert index._is_strictly_monotonic_decreasing is True
        index = index_cls([1])
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_decreasing is True
        assert index._is_strictly_monotonic_increasing is True
        assert index._is_strictly_monotonic_decreasing is True

    def test_is_strictly_monotonic(self):
        index_cls = Index
        index = index_cls([1, 1, 2, 3])
        assert index.is_monotonic_increasing is True
        assert index._is_strictly_monotonic_increasing is False
        index = index_cls([3, 2, 1, 1])
        assert index.is_monotonic_decreasing is True
        assert index._is_strictly_monotonic_decreasing is False
        index = index_cls([1, 1])
        assert index.is_monotonic_increasing
        assert index.is_monotonic_decreasing
        assert not index._is_strictly_monotonic_increasing
        assert not index._is_strictly_monotonic_decreasing

    def test_logical_compat(self, simple_index):
        idx = simple_index
        assert idx.all() == idx.values.all()
        assert idx.any() == idx.values.any()

    def test_identical(self, simple_index, dtype):
        index = simple_index
        idx = Index(index.copy())
        assert idx.identical(index)
        same_values_different_type = Index(idx, dtype=object)
        assert not idx.identical(same_values_different_type)
        idx = index.astype(dtype=object)
        idx = idx.rename('foo')
        same_values = Index(idx, dtype=object)
        assert same_values.identical(idx)
        assert not idx.identical(index)
        assert Index(same_values, name='foo', dtype=object).identical(idx)
        assert not index.astype(dtype=object).identical(index.astype(dtype=dtype))

    def test_cant_or_shouldnt_cast(self, dtype):
        msg = "invalid literal for int\\(\\) with base 10: 'foo'"
        data = ['foo', 'bar', 'baz']
        with pytest.raises(ValueError, match=msg):
            Index(data, dtype=dtype)

    def test_view_index(self, simple_index):
        index = simple_index
        msg = 'Passing a type in .*Index.view is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            index.view(Index)

    def test_prevent_casting(self, simple_index):
        index = simple_index
        result = index.astype('O')
        assert result.dtype == np.object_