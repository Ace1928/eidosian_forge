from __future__ import annotations
from datetime import datetime
import gc
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BaseMaskedArray
class NumericBase(Base):
    """
    Base class for numeric index (incl. RangeIndex) sub-class tests.
    """

    def test_constructor_unwraps_index(self, dtype):
        index_cls = self._index_cls
        idx = Index([1, 2], dtype=dtype)
        result = index_cls(idx)
        expected = np.array([1, 2], dtype=idx.dtype)
        tm.assert_numpy_array_equal(result._data, expected)

    def test_where(self):
        pass

    def test_can_hold_identifiers(self, simple_index):
        idx = simple_index
        key = idx[0]
        assert idx._can_hold_identifiers_and_holds_name(key) is False

    def test_view(self, dtype):
        index_cls = self._index_cls
        idx = index_cls([], dtype=dtype, name='Foo')
        idx_view = idx.view()
        assert idx_view.name == 'Foo'
        idx_view = idx.view(dtype)
        tm.assert_index_equal(idx, index_cls(idx_view, name='Foo'), exact=True)
        idx_view = idx.view(index_cls)
        tm.assert_index_equal(idx, index_cls(idx_view, name='Foo'), exact=True)

    def test_format(self, simple_index):
        idx = simple_index
        max_width = max((len(str(x)) for x in idx))
        expected = [str(x).ljust(max_width) for x in idx]
        assert idx.format() == expected

    def test_numeric_compat(self):
        pass

    def test_insert_non_na(self, simple_index):
        index = simple_index
        result = index.insert(0, index[0])
        expected = Index([index[0]] + list(index), dtype=index.dtype)
        tm.assert_index_equal(result, expected, exact=True)

    def test_insert_na(self, nulls_fixture, simple_index):
        index = simple_index
        na_val = nulls_fixture
        if na_val is pd.NaT:
            expected = Index([index[0], pd.NaT] + list(index[1:]), dtype=object)
        else:
            expected = Index([index[0], np.nan] + list(index[1:]))
            if index.dtype.kind == 'f':
                expected = Index(expected, dtype=index.dtype)
        result = index.insert(1, na_val)
        tm.assert_index_equal(result, expected, exact=True)

    def test_arithmetic_explicit_conversions(self):
        index_cls = self._index_cls
        if index_cls is RangeIndex:
            idx = RangeIndex(5)
        else:
            idx = index_cls(np.arange(5, dtype='int64'))
        arr = np.arange(5, dtype='int64') * 3.2
        expected = Index(arr, dtype=np.float64)
        fidx = idx * 3.2
        tm.assert_index_equal(fidx, expected)
        fidx = 3.2 * idx
        tm.assert_index_equal(fidx, expected)
        expected = Index(arr, dtype=np.float64)
        a = np.zeros(5, dtype='float64')
        result = fidx - a
        tm.assert_index_equal(result, expected)
        expected = Index(-arr, dtype=np.float64)
        a = np.zeros(5, dtype='float64')
        result = a - fidx
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('complex_dtype', [np.complex64, np.complex128])
    def test_astype_to_complex(self, complex_dtype, simple_index):
        result = simple_index.astype(complex_dtype)
        assert type(result) is Index and result.dtype == complex_dtype

    def test_cast_string(self, dtype):
        result = self._index_cls(['0', '1', '2'], dtype=dtype)
        expected = self._index_cls([0, 1, 2], dtype=dtype)
        tm.assert_index_equal(result, expected)