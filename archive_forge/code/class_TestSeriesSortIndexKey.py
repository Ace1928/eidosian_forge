import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
class TestSeriesSortIndexKey:

    def test_sort_index_multiindex_key(self):
        mi = MultiIndex.from_tuples([[1, 1, 3], [1, 1, 1]], names=list('ABC'))
        s = Series([1, 2], mi)
        backwards = s.iloc[[1, 0]]
        result = s.sort_index(level='C', key=lambda x: -x)
        tm.assert_series_equal(s, result)
        result = s.sort_index(level='C', key=lambda x: x)
        tm.assert_series_equal(backwards, result)

    def test_sort_index_multiindex_key_multi_level(self):
        mi = MultiIndex.from_tuples([[1, 1, 3], [1, 1, 1]], names=list('ABC'))
        s = Series([1, 2], mi)
        backwards = s.iloc[[1, 0]]
        result = s.sort_index(level=['A', 'C'], key=lambda x: -x)
        tm.assert_series_equal(s, result)
        result = s.sort_index(level=['A', 'C'], key=lambda x: x)
        tm.assert_series_equal(backwards, result)

    def test_sort_index_key(self):
        series = Series(np.arange(6, dtype='int64'), index=list('aaBBca'))
        result = series.sort_index()
        expected = series.iloc[[2, 3, 0, 1, 5, 4]]
        tm.assert_series_equal(result, expected)
        result = series.sort_index(key=lambda x: x.str.lower())
        expected = series.iloc[[0, 1, 5, 2, 3, 4]]
        tm.assert_series_equal(result, expected)
        result = series.sort_index(key=lambda x: x.str.lower(), ascending=False)
        expected = series.iloc[[4, 2, 3, 0, 1, 5]]
        tm.assert_series_equal(result, expected)

    def test_sort_index_key_int(self):
        series = Series(np.arange(6, dtype='int64'), index=np.arange(6, dtype='int64'))
        result = series.sort_index()
        tm.assert_series_equal(result, series)
        result = series.sort_index(key=lambda x: -x)
        expected = series.sort_index(ascending=False)
        tm.assert_series_equal(result, expected)
        result = series.sort_index(key=lambda x: 2 * x)
        tm.assert_series_equal(result, series)

    def test_sort_index_kind_key(self, sort_kind, sort_by_key):
        series = Series(index=[3, 2, 1, 4, 3], dtype=object)
        expected_series = Series(index=[1, 2, 3, 3, 4], dtype=object)
        index_sorted_series = series.sort_index(kind=sort_kind, key=sort_by_key)
        tm.assert_series_equal(expected_series, index_sorted_series)

    def test_sort_index_kind_neg_key(self, sort_kind):
        series = Series(index=[3, 2, 1, 4, 3], dtype=object)
        expected_series = Series(index=[4, 3, 3, 2, 1], dtype=object)
        index_sorted_series = series.sort_index(kind=sort_kind, key=lambda x: -x)
        tm.assert_series_equal(expected_series, index_sorted_series)

    def test_sort_index_na_position_key(self, sort_by_key):
        series = Series(index=[3, 2, 1, 4, 3, np.nan], dtype=object)
        expected_series_first = Series(index=[np.nan, 1, 2, 3, 3, 4], dtype=object)
        index_sorted_series = series.sort_index(na_position='first', key=sort_by_key)
        tm.assert_series_equal(expected_series_first, index_sorted_series)
        expected_series_last = Series(index=[1, 2, 3, 3, 4, np.nan], dtype=object)
        index_sorted_series = series.sort_index(na_position='last', key=sort_by_key)
        tm.assert_series_equal(expected_series_last, index_sorted_series)

    def test_changes_length_raises(self):
        s = Series([1, 2, 3])
        with pytest.raises(ValueError, match='change the shape'):
            s.sort_index(key=lambda x: x[:1])

    def test_sort_values_key_type(self):
        s = Series([1, 2, 3], DatetimeIndex(['2008-10-24', '2008-11-23', '2007-12-22']))
        result = s.sort_index(key=lambda x: x.month)
        expected = s.iloc[[0, 1, 2]]
        tm.assert_series_equal(result, expected)
        result = s.sort_index(key=lambda x: x.day)
        expected = s.iloc[[2, 1, 0]]
        tm.assert_series_equal(result, expected)
        result = s.sort_index(key=lambda x: x.year)
        expected = s.iloc[[2, 0, 1]]
        tm.assert_series_equal(result, expected)
        result = s.sort_index(key=lambda x: x.month_name())
        expected = s.iloc[[2, 1, 0]]
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('ascending', [[True, False], [False, True]])
    def test_sort_index_multi_already_monotonic(self, ascending):
        mi = MultiIndex.from_product([[1, 2], [3, 4]])
        ser = Series(range(len(mi)), index=mi)
        result = ser.sort_index(ascending=ascending)
        if ascending == [True, False]:
            expected = ser.take([1, 0, 3, 2])
        elif ascending == [False, True]:
            expected = ser.take([2, 3, 0, 1])
        tm.assert_series_equal(result, expected)