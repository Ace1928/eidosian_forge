import re
import sys
import numpy as np
import pytest
from pandas.compat import PYPY
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
class TestCategoricalAnalytics:

    @pytest.mark.parametrize('aggregation', ['min', 'max'])
    def test_min_max_not_ordered_raises(self, aggregation):
        cat = Categorical(['a', 'b', 'c', 'd'], ordered=False)
        msg = f'Categorical is not ordered for operation {aggregation}'
        agg_func = getattr(cat, aggregation)
        with pytest.raises(TypeError, match=msg):
            agg_func()
        ufunc = np.minimum if aggregation == 'min' else np.maximum
        with pytest.raises(TypeError, match=msg):
            ufunc.reduce(cat)

    def test_min_max_ordered(self, index_or_series_or_array):
        cat = Categorical(['a', 'b', 'c', 'd'], ordered=True)
        obj = index_or_series_or_array(cat)
        _min = obj.min()
        _max = obj.max()
        assert _min == 'a'
        assert _max == 'd'
        assert np.minimum.reduce(obj) == 'a'
        assert np.maximum.reduce(obj) == 'd'
        cat = Categorical(['a', 'b', 'c', 'd'], categories=['d', 'c', 'b', 'a'], ordered=True)
        obj = index_or_series_or_array(cat)
        _min = obj.min()
        _max = obj.max()
        assert _min == 'd'
        assert _max == 'a'
        assert np.minimum.reduce(obj) == 'd'
        assert np.maximum.reduce(obj) == 'a'

    def test_min_max_reduce(self):
        cat = Categorical(['a', 'b', 'c', 'd'], ordered=True)
        df = DataFrame(cat)
        result_max = df.agg('max')
        expected_max = Series(Categorical(['d'], dtype=cat.dtype))
        tm.assert_series_equal(result_max, expected_max)
        result_min = df.agg('min')
        expected_min = Series(Categorical(['a'], dtype=cat.dtype))
        tm.assert_series_equal(result_min, expected_min)

    @pytest.mark.parametrize('categories,expected', [(list('ABC'), np.nan), ([1, 2, 3], np.nan), pytest.param(Series(date_range('2020-01-01', periods=3), dtype='category'), NaT, marks=pytest.mark.xfail(reason='https://github.com/pandas-dev/pandas/issues/29962'))])
    @pytest.mark.parametrize('aggregation', ['min', 'max'])
    def test_min_max_ordered_empty(self, categories, expected, aggregation):
        cat = Categorical([], categories=categories, ordered=True)
        agg_func = getattr(cat, aggregation)
        result = agg_func()
        assert result is expected

    @pytest.mark.parametrize('values, categories', [(['a', 'b', 'c', np.nan], list('cba')), ([1, 2, 3, np.nan], [3, 2, 1])])
    @pytest.mark.parametrize('skipna', [True, False])
    @pytest.mark.parametrize('function', ['min', 'max'])
    def test_min_max_with_nan(self, values, categories, function, skipna):
        cat = Categorical(values, categories=categories, ordered=True)
        result = getattr(cat, function)(skipna=skipna)
        if skipna is False:
            assert result is np.nan
        else:
            expected = categories[0] if function == 'min' else categories[2]
            assert result == expected

    @pytest.mark.parametrize('function', ['min', 'max'])
    @pytest.mark.parametrize('skipna', [True, False])
    def test_min_max_only_nan(self, function, skipna):
        cat = Categorical([np.nan], categories=[1, 2], ordered=True)
        result = getattr(cat, function)(skipna=skipna)
        assert result is np.nan

    @pytest.mark.parametrize('method', ['min', 'max'])
    def test_numeric_only_min_max_raises(self, method):
        cat = Categorical([np.nan, 1, 2, np.nan], categories=[5, 4, 3, 2, 1], ordered=True)
        with pytest.raises(TypeError, match='.* got an unexpected keyword'):
            getattr(cat, method)(numeric_only=True)

    @pytest.mark.parametrize('method', ['min', 'max'])
    def test_numpy_min_max_raises(self, method):
        cat = Categorical(['a', 'b', 'c', 'b'], ordered=False)
        msg = f'Categorical is not ordered for operation {method}\nyou can use .as_ordered() to change the Categorical to an ordered one'
        method = getattr(np, method)
        with pytest.raises(TypeError, match=re.escape(msg)):
            method(cat)

    @pytest.mark.parametrize('kwarg', ['axis', 'out', 'keepdims'])
    @pytest.mark.parametrize('method', ['min', 'max'])
    def test_numpy_min_max_unsupported_kwargs_raises(self, method, kwarg):
        cat = Categorical(['a', 'b', 'c', 'b'], ordered=True)
        msg = f"the '{kwarg}' parameter is not supported in the pandas implementation of {method}"
        if kwarg == 'axis':
            msg = '`axis` must be fewer than the number of dimensions \\(1\\)'
        kwargs = {kwarg: 42}
        method = getattr(np, method)
        with pytest.raises(ValueError, match=msg):
            method(cat, **kwargs)

    @pytest.mark.parametrize('method, expected', [('min', 'a'), ('max', 'c')])
    def test_numpy_min_max_axis_equals_none(self, method, expected):
        cat = Categorical(['a', 'b', 'c', 'b'], ordered=True)
        method = getattr(np, method)
        result = method(cat, axis=None)
        assert result == expected

    @pytest.mark.parametrize('values,categories,exp_mode', [([1, 1, 2, 4, 5, 5, 5], [5, 4, 3, 2, 1], [5]), ([1, 1, 1, 4, 5, 5, 5], [5, 4, 3, 2, 1], [5, 1]), ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [5, 4, 3, 2, 1]), ([np.nan, np.nan, np.nan, 4, 5], [5, 4, 3, 2, 1], [5, 4]), ([np.nan, np.nan, np.nan, 4, 5, 4], [5, 4, 3, 2, 1], [4]), ([np.nan, np.nan, 4, 5, 4], [5, 4, 3, 2, 1], [4])])
    def test_mode(self, values, categories, exp_mode):
        cat = Categorical(values, categories=categories, ordered=True)
        res = Series(cat).mode()._values
        exp = Categorical(exp_mode, categories=categories, ordered=True)
        tm.assert_categorical_equal(res, exp)

    def test_searchsorted(self, ordered):
        cat = Categorical(['cheese', 'milk', 'apple', 'bread', 'bread'], categories=['cheese', 'milk', 'apple', 'bread'], ordered=ordered)
        ser = Series(cat)
        res_cat = cat.searchsorted('apple')
        assert res_cat == 2
        assert is_scalar(res_cat)
        res_ser = ser.searchsorted('apple')
        assert res_ser == 2
        assert is_scalar(res_ser)
        res_cat = cat.searchsorted(['bread'])
        res_ser = ser.searchsorted(['bread'])
        exp = np.array([3], dtype=np.intp)
        tm.assert_numpy_array_equal(res_cat, exp)
        tm.assert_numpy_array_equal(res_ser, exp)
        res_cat = cat.searchsorted(['apple', 'bread'], side='right')
        res_ser = ser.searchsorted(['apple', 'bread'], side='right')
        exp = np.array([3, 5], dtype=np.intp)
        tm.assert_numpy_array_equal(res_cat, exp)
        tm.assert_numpy_array_equal(res_ser, exp)
        with pytest.raises(TypeError, match='cucumber'):
            cat.searchsorted('cucumber')
        with pytest.raises(TypeError, match='cucumber'):
            ser.searchsorted('cucumber')
        msg = 'Cannot setitem on a Categorical with a new category, set the categories first'
        with pytest.raises(TypeError, match=msg):
            cat.searchsorted(['bread', 'cucumber'])
        with pytest.raises(TypeError, match=msg):
            ser.searchsorted(['bread', 'cucumber'])

    def test_unique(self, ordered):
        dtype = CategoricalDtype(['a', 'b', 'c'], ordered=ordered)
        cat = Categorical(['a', 'b', 'c'], dtype=dtype)
        res = cat.unique()
        tm.assert_categorical_equal(res, cat)
        cat = Categorical(['a', 'b', 'a', 'a'], dtype=dtype)
        res = cat.unique()
        tm.assert_categorical_equal(res, Categorical(['a', 'b'], dtype=dtype))
        cat = Categorical(['c', 'a', 'b', 'a', 'a'], dtype=dtype)
        res = cat.unique()
        exp_cat = Categorical(['c', 'a', 'b'], dtype=dtype)
        tm.assert_categorical_equal(res, exp_cat)
        cat = Categorical(['b', np.nan, 'b', np.nan, 'a'], dtype=dtype)
        res = cat.unique()
        exp_cat = Categorical(['b', np.nan, 'a'], dtype=dtype)
        tm.assert_categorical_equal(res, exp_cat)

    def test_unique_index_series(self, ordered):
        dtype = CategoricalDtype([3, 2, 1], ordered=ordered)
        c = Categorical([3, 1, 2, 2, 1], dtype=dtype)
        exp = Categorical([3, 1, 2], dtype=dtype)
        tm.assert_categorical_equal(c.unique(), exp)
        tm.assert_index_equal(Index(c).unique(), Index(exp))
        tm.assert_categorical_equal(Series(c).unique(), exp)
        c = Categorical([1, 1, 2, 2], dtype=dtype)
        exp = Categorical([1, 2], dtype=dtype)
        tm.assert_categorical_equal(c.unique(), exp)
        tm.assert_index_equal(Index(c).unique(), Index(exp))
        tm.assert_categorical_equal(Series(c).unique(), exp)

    def test_shift(self):
        cat = Categorical(['a', 'b', 'c', 'd', 'a'])
        sp1 = cat.shift(1)
        xp1 = Categorical([np.nan, 'a', 'b', 'c', 'd'])
        tm.assert_categorical_equal(sp1, xp1)
        tm.assert_categorical_equal(cat[:-1], sp1[1:])
        sn2 = cat.shift(-2)
        xp2 = Categorical(['c', 'd', 'a', np.nan, np.nan], categories=['a', 'b', 'c', 'd'])
        tm.assert_categorical_equal(sn2, xp2)
        tm.assert_categorical_equal(cat[2:], sn2[:-2])
        tm.assert_categorical_equal(cat, cat.shift(0))

    def test_nbytes(self):
        cat = Categorical([1, 2, 3])
        exp = 3 + 3 * 8
        assert cat.nbytes == exp

    def test_memory_usage(self):
        cat = Categorical([1, 2, 3])
        assert 0 < cat.nbytes <= cat.memory_usage()
        assert 0 < cat.nbytes <= cat.memory_usage(deep=True)
        cat = Categorical(['foo', 'foo', 'bar'])
        assert cat.memory_usage(deep=True) > cat.nbytes
        if not PYPY:
            diff = cat.memory_usage(deep=True) - sys.getsizeof(cat)
            assert abs(diff) < 100

    def test_map(self):
        c = Categorical(list('ABABC'), categories=list('CBA'), ordered=True)
        result = c.map(lambda x: x.lower(), na_action=None)
        exp = Categorical(list('ababc'), categories=list('cba'), ordered=True)
        tm.assert_categorical_equal(result, exp)
        c = Categorical(list('ABABC'), categories=list('ABC'), ordered=False)
        result = c.map(lambda x: x.lower(), na_action=None)
        exp = Categorical(list('ababc'), categories=list('abc'), ordered=False)
        tm.assert_categorical_equal(result, exp)
        result = c.map(lambda x: 1, na_action=None)
        tm.assert_index_equal(result, Index(np.array([1] * 5, dtype=np.int64)))

    @pytest.mark.parametrize('value', [1, 'True', [1, 2, 3], 5.0])
    def test_validate_inplace_raises(self, value):
        cat = Categorical(['A', 'B', 'B', 'C', 'A'])
        msg = f'For argument "inplace" expected type bool, received type {type(value).__name__}'
        with pytest.raises(ValueError, match=msg):
            cat.sort_values(inplace=value)

    def test_quantile_empty(self):
        cat = Categorical(['A', 'B'])
        idx = Index([0.0, 0.5])
        result = cat[:0]._quantile(idx, interpolation='linear')
        assert result._codes.dtype == np.int8
        expected = cat.take([-1, -1], allow_fill=True)
        tm.assert_extension_array_equal(result, expected)