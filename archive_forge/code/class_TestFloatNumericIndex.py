import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestFloatNumericIndex:

    @pytest.fixture(params=[np.float64, np.float32])
    def dtype(self, request):
        return request.param

    @pytest.fixture
    def simple_index(self, dtype):
        values = np.arange(5, dtype=dtype)
        return Index(values)

    @pytest.fixture(params=[[1.5, 2, 3, 4, 5], [0.0, 2.5, 5.0, 7.5, 10.0], [5, 4, 3, 2, 1.5], [10.0, 7.5, 5.0, 2.5, 0.0]], ids=['mixed', 'float', 'mixed_dec', 'float_dec'])
    def index(self, request, dtype):
        return Index(request.param, dtype=dtype)

    @pytest.fixture
    def mixed_index(self, dtype):
        return Index([1.5, 2, 3, 4, 5], dtype=dtype)

    @pytest.fixture
    def float_index(self, dtype):
        return Index([0.0, 2.5, 5.0, 7.5, 10.0], dtype=dtype)

    def test_repr_roundtrip(self, index):
        tm.assert_index_equal(eval(repr(index)), index, exact=True)

    def check_coerce(self, a, b, is_float_index=True):
        assert a.equals(b)
        tm.assert_index_equal(a, b, exact=False)
        if is_float_index:
            assert isinstance(b, Index)
        else:
            assert type(b) is Index

    def test_constructor_from_list_no_dtype(self):
        index = Index([1.5, 2.5, 3.5])
        assert index.dtype == np.float64

    def test_constructor(self, dtype):
        index_cls = Index
        index = index_cls([1, 2, 3, 4, 5], dtype=dtype)
        assert isinstance(index, index_cls)
        assert index.dtype == dtype
        expected = np.array([1, 2, 3, 4, 5], dtype=dtype)
        tm.assert_numpy_array_equal(index.values, expected)
        index = index_cls(np.array([1, 2, 3, 4, 5]), dtype=dtype)
        assert isinstance(index, index_cls)
        assert index.dtype == dtype
        index = index_cls([1.0, 2, 3, 4, 5], dtype=dtype)
        assert isinstance(index, index_cls)
        assert index.dtype == dtype
        index = index_cls(np.array([1.0, 2, 3, 4, 5]), dtype=dtype)
        assert isinstance(index, index_cls)
        assert index.dtype == dtype
        index = index_cls([1.0, 2, 3, 4, 5], dtype=dtype)
        assert isinstance(index, index_cls)
        assert index.dtype == dtype
        index = index_cls(np.array([1.0, 2, 3, 4, 5]), dtype=dtype)
        assert isinstance(index, index_cls)
        assert index.dtype == dtype
        result = index_cls([np.nan, np.nan], dtype=dtype)
        assert pd.isna(result.values).all()
        result = index_cls(np.array([np.nan]), dtype=dtype)
        assert pd.isna(result.values).all()

    def test_constructor_invalid(self):
        index_cls = Index
        cls_name = index_cls.__name__
        msg = f'{cls_name}\\(\\.\\.\\.\\) must be called with a collection of some kind, 0\\.0 was passed'
        with pytest.raises(TypeError, match=msg):
            index_cls(0.0)

    def test_constructor_coerce(self, mixed_index, float_index):
        self.check_coerce(mixed_index, Index([1.5, 2, 3, 4, 5]))
        self.check_coerce(float_index, Index(np.arange(5) * 2.5))
        result = Index(np.array(np.arange(5) * 2.5, dtype=object))
        assert result.dtype == object
        self.check_coerce(float_index, result.astype('float64'))

    def test_constructor_explicit(self, mixed_index, float_index):
        self.check_coerce(float_index, Index(np.arange(5) * 2.5, dtype=object), is_float_index=False)
        self.check_coerce(mixed_index, Index([1.5, 2, 3, 4, 5], dtype=object), is_float_index=False)

    def test_type_coercion_fail(self, any_int_numpy_dtype):
        msg = 'Trying to coerce float values to integers'
        with pytest.raises(ValueError, match=msg):
            Index([1, 2, 3.5], dtype=any_int_numpy_dtype)

    def test_equals_numeric(self):
        index_cls = Index
        idx = index_cls([1.0, 2.0])
        assert idx.equals(idx)
        assert idx.identical(idx)
        idx2 = index_cls([1.0, 2.0])
        assert idx.equals(idx2)
        idx = index_cls([1.0, np.nan])
        assert idx.equals(idx)
        assert idx.identical(idx)
        idx2 = index_cls([1.0, np.nan])
        assert idx.equals(idx2)

    @pytest.mark.parametrize('other', (Index([1, 2], dtype=np.int64), Index([1.0, 2.0], dtype=object), Index([1, 2], dtype=object)))
    def test_equals_numeric_other_index_type(self, other):
        idx = Index([1.0, 2.0])
        assert idx.equals(other)
        assert other.equals(idx)

    @pytest.mark.parametrize('vals', [pd.date_range('2016-01-01', periods=3), pd.timedelta_range('1 Day', periods=3)])
    def test_lookups_datetimelike_values(self, vals, dtype):
        ser = Series(vals, index=range(3, 6))
        ser.index = ser.index.astype(dtype)
        expected = vals[1]
        result = ser[4.0]
        assert isinstance(result, type(expected)) and result == expected
        result = ser[4]
        assert isinstance(result, type(expected)) and result == expected
        result = ser.loc[4.0]
        assert isinstance(result, type(expected)) and result == expected
        result = ser.loc[4]
        assert isinstance(result, type(expected)) and result == expected
        result = ser.at[4.0]
        assert isinstance(result, type(expected)) and result == expected
        result = ser.at[4]
        assert isinstance(result, type(expected)) and result == expected
        result = ser.iloc[1]
        assert isinstance(result, type(expected)) and result == expected
        result = ser.iat[1]
        assert isinstance(result, type(expected)) and result == expected

    def test_doesnt_contain_all_the_things(self):
        idx = Index([np.nan])
        assert not idx.isin([0]).item()
        assert not idx.isin([1]).item()
        assert idx.isin([np.nan]).item()

    def test_nan_multiple_containment(self):
        index_cls = Index
        idx = index_cls([1.0, np.nan])
        tm.assert_numpy_array_equal(idx.isin([1.0]), np.array([True, False]))
        tm.assert_numpy_array_equal(idx.isin([2.0, np.pi]), np.array([False, False]))
        tm.assert_numpy_array_equal(idx.isin([np.nan]), np.array([False, True]))
        tm.assert_numpy_array_equal(idx.isin([1.0, np.nan]), np.array([True, True]))
        idx = index_cls([1.0, 2.0])
        tm.assert_numpy_array_equal(idx.isin([np.nan]), np.array([False, False]))

    def test_fillna_float64(self):
        index_cls = Index
        idx = Index([1.0, np.nan, 3.0], dtype=float, name='x')
        exp = Index([1.0, 0.1, 3.0], name='x')
        tm.assert_index_equal(idx.fillna(0.1), exp, exact=True)
        exp = index_cls([1.0, 2.0, 3.0], name='x')
        tm.assert_index_equal(idx.fillna(2), exp)
        exp = Index([1.0, 'obj', 3.0], name='x')
        tm.assert_index_equal(idx.fillna('obj'), exp, exact=True)

    def test_logical_compat(self, simple_index):
        idx = simple_index
        assert idx.all() == idx.values.all()
        assert idx.any() == idx.values.any()
        assert idx.all() == idx.to_series().all()
        assert idx.any() == idx.to_series().any()