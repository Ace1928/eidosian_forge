import math
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
class TestCategoricalIndexing:

    def test_getitem_slice(self):
        cat = Categorical(['a', 'b', 'c', 'd', 'a', 'b', 'c'])
        sliced = cat[3]
        assert sliced == 'd'
        sliced = cat[3:5]
        expected = Categorical(['d', 'a'], categories=['a', 'b', 'c', 'd'])
        tm.assert_categorical_equal(sliced, expected)

    def test_getitem_listlike(self):
        c = Categorical(np.random.default_rng(2).integers(0, 5, size=150000).astype(np.int8))
        result = c.codes[np.array([100000]).astype(np.int64)]
        expected = c[np.array([100000]).astype(np.int64)].codes
        tm.assert_numpy_array_equal(result, expected)

    def test_periodindex(self):
        idx1 = PeriodIndex(['2014-01', '2014-01', '2014-02', '2014-02', '2014-03', '2014-03'], freq='M')
        cat1 = Categorical(idx1)
        str(cat1)
        exp_arr = np.array([0, 0, 1, 1, 2, 2], dtype=np.int8)
        exp_idx = PeriodIndex(['2014-01', '2014-02', '2014-03'], freq='M')
        tm.assert_numpy_array_equal(cat1._codes, exp_arr)
        tm.assert_index_equal(cat1.categories, exp_idx)
        idx2 = PeriodIndex(['2014-03', '2014-03', '2014-02', '2014-01', '2014-03', '2014-01'], freq='M')
        cat2 = Categorical(idx2, ordered=True)
        str(cat2)
        exp_arr = np.array([2, 2, 1, 0, 2, 0], dtype=np.int8)
        exp_idx2 = PeriodIndex(['2014-01', '2014-02', '2014-03'], freq='M')
        tm.assert_numpy_array_equal(cat2._codes, exp_arr)
        tm.assert_index_equal(cat2.categories, exp_idx2)
        idx3 = PeriodIndex(['2013-12', '2013-11', '2013-10', '2013-09', '2013-08', '2013-07', '2013-05'], freq='M')
        cat3 = Categorical(idx3, ordered=True)
        exp_arr = np.array([6, 5, 4, 3, 2, 1, 0], dtype=np.int8)
        exp_idx = PeriodIndex(['2013-05', '2013-07', '2013-08', '2013-09', '2013-10', '2013-11', '2013-12'], freq='M')
        tm.assert_numpy_array_equal(cat3._codes, exp_arr)
        tm.assert_index_equal(cat3.categories, exp_idx)

    @pytest.mark.parametrize('null_val', [None, np.nan, NaT, NA, math.nan, 'NaT', 'nat', 'NAT', 'nan', 'NaN', 'NAN'])
    def test_periodindex_on_null_types(self, null_val):
        result = PeriodIndex(['2022-04-06', '2022-04-07', null_val], freq='D')
        expected = PeriodIndex(['2022-04-06', '2022-04-07', 'NaT'], dtype='period[D]')
        assert result[2] is NaT
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('new_categories', [[1, 2, 3, 4], [1, 2]])
    def test_categories_assignments_wrong_length_raises(self, new_categories):
        cat = Categorical(['a', 'b', 'c', 'a'])
        msg = 'new categories need to have the same number of items as the old categories!'
        with pytest.raises(ValueError, match=msg):
            cat.rename_categories(new_categories)

    @pytest.mark.parametrize('idx_values', [[1, 2, 3, 4], [1, 3, 2, 4], [1, 3, 3, 4], [1, 2, 2, 4]])
    @pytest.mark.parametrize('key_values', [[1, 2], [1, 5], [1, 1], [5, 5]])
    @pytest.mark.parametrize('key_class', [Categorical, CategoricalIndex])
    @pytest.mark.parametrize('dtype', [None, 'category', 'key'])
    def test_get_indexer_non_unique(self, idx_values, key_values, key_class, dtype):
        key = key_class(key_values, categories=range(1, 5))
        if dtype == 'key':
            dtype = key.dtype
        idx = Index(idx_values, dtype=dtype)
        expected, exp_miss = idx.get_indexer_non_unique(key_values)
        result, res_miss = idx.get_indexer_non_unique(key)
        tm.assert_numpy_array_equal(expected, result)
        tm.assert_numpy_array_equal(exp_miss, res_miss)
        exp_unique = idx.unique().get_indexer(key_values)
        res_unique = idx.unique().get_indexer(key)
        tm.assert_numpy_array_equal(res_unique, exp_unique)

    def test_where_unobserved_nan(self):
        ser = Series(Categorical(['a', 'b']))
        result = ser.where([True, False])
        expected = Series(Categorical(['a', None], categories=['a', 'b']))
        tm.assert_series_equal(result, expected)
        ser = Series(Categorical(['a', 'b']))
        result = ser.where([False, False])
        expected = Series(Categorical([None, None], categories=['a', 'b']))
        tm.assert_series_equal(result, expected)

    def test_where_unobserved_categories(self):
        ser = Series(Categorical(['a', 'b', 'c'], categories=['d', 'c', 'b', 'a']))
        result = ser.where([True, True, False], other='b')
        expected = Series(Categorical(['a', 'b', 'b'], categories=ser.cat.categories))
        tm.assert_series_equal(result, expected)

    def test_where_other_categorical(self):
        ser = Series(Categorical(['a', 'b', 'c'], categories=['d', 'c', 'b', 'a']))
        other = Categorical(['b', 'c', 'a'], categories=['a', 'c', 'b', 'd'])
        result = ser.where([True, False, True], other)
        expected = Series(Categorical(['a', 'c', 'c'], dtype=ser.dtype))
        tm.assert_series_equal(result, expected)

    def test_where_new_category_raises(self):
        ser = Series(Categorical(['a', 'b', 'c']))
        msg = 'Cannot setitem on a Categorical with a new category'
        with pytest.raises(TypeError, match=msg):
            ser.where([True, False, True], 'd')

    def test_where_ordered_differs_rasies(self):
        ser = Series(Categorical(['a', 'b', 'c'], categories=['d', 'c', 'b', 'a'], ordered=True))
        other = Categorical(['b', 'c', 'a'], categories=['a', 'c', 'b', 'd'], ordered=True)
        with pytest.raises(TypeError, match='without identical categories'):
            ser.where([True, False, True], other)