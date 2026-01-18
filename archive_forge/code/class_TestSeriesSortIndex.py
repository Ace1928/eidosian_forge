import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
class TestSeriesSortIndex:

    def test_sort_index_name(self, datetime_series):
        result = datetime_series.sort_index(ascending=False)
        assert result.name == datetime_series.name

    def test_sort_index(self, datetime_series):
        datetime_series.index = datetime_series.index._with_freq(None)
        rindex = list(datetime_series.index)
        np.random.default_rng(2).shuffle(rindex)
        random_order = datetime_series.reindex(rindex)
        sorted_series = random_order.sort_index()
        tm.assert_series_equal(sorted_series, datetime_series)
        sorted_series = random_order.sort_index(ascending=False)
        tm.assert_series_equal(sorted_series, datetime_series.reindex(datetime_series.index[::-1]))
        sorted_series = random_order.sort_index(level=0)
        tm.assert_series_equal(sorted_series, datetime_series)
        sorted_series = random_order.sort_index(axis=0)
        tm.assert_series_equal(sorted_series, datetime_series)
        msg = 'No axis named 1 for object type Series'
        with pytest.raises(ValueError, match=msg):
            random_order.sort_values(axis=1)
        sorted_series = random_order.sort_index(level=0, axis=0)
        tm.assert_series_equal(sorted_series, datetime_series)
        with pytest.raises(ValueError, match=msg):
            random_order.sort_index(level=0, axis=1)

    def test_sort_index_inplace(self, datetime_series):
        datetime_series.index = datetime_series.index._with_freq(None)
        rindex = list(datetime_series.index)
        np.random.default_rng(2).shuffle(rindex)
        random_order = datetime_series.reindex(rindex)
        result = random_order.sort_index(ascending=False, inplace=True)
        assert result is None
        expected = datetime_series.reindex(datetime_series.index[::-1])
        expected.index = expected.index._with_freq(None)
        tm.assert_series_equal(random_order, expected)
        random_order = datetime_series.reindex(rindex)
        result = random_order.sort_index(ascending=True, inplace=True)
        assert result is None
        expected = datetime_series.copy()
        expected.index = expected.index._with_freq(None)
        tm.assert_series_equal(random_order, expected)

    def test_sort_index_level(self):
        mi = MultiIndex.from_tuples([[1, 1, 3], [1, 1, 1]], names=list('ABC'))
        s = Series([1, 2], mi)
        backwards = s.iloc[[1, 0]]
        res = s.sort_index(level='A')
        tm.assert_series_equal(backwards, res)
        res = s.sort_index(level=['A', 'B'])
        tm.assert_series_equal(backwards, res)
        res = s.sort_index(level='A', sort_remaining=False)
        tm.assert_series_equal(s, res)
        res = s.sort_index(level=['A', 'B'], sort_remaining=False)
        tm.assert_series_equal(s, res)

    @pytest.mark.parametrize('level', ['A', 0])
    def test_sort_index_multiindex(self, level):
        mi = MultiIndex.from_tuples([[1, 1, 3], [1, 1, 1]], names=list('ABC'))
        s = Series([1, 2], mi)
        backwards = s.iloc[[1, 0]]
        res = s.sort_index(level=level)
        tm.assert_series_equal(backwards, res)
        res = s.sort_index(level=level, sort_remaining=False)
        tm.assert_series_equal(s, res)

    def test_sort_index_kind(self, sort_kind):
        series = Series(index=[3, 2, 1, 4, 3], dtype=object)
        expected_series = Series(index=[1, 2, 3, 3, 4], dtype=object)
        index_sorted_series = series.sort_index(kind=sort_kind)
        tm.assert_series_equal(expected_series, index_sorted_series)

    def test_sort_index_na_position(self):
        series = Series(index=[3, 2, 1, 4, 3, np.nan], dtype=object)
        expected_series_first = Series(index=[np.nan, 1, 2, 3, 3, 4], dtype=object)
        index_sorted_series = series.sort_index(na_position='first')
        tm.assert_series_equal(expected_series_first, index_sorted_series)
        expected_series_last = Series(index=[1, 2, 3, 3, 4, np.nan], dtype=object)
        index_sorted_series = series.sort_index(na_position='last')
        tm.assert_series_equal(expected_series_last, index_sorted_series)

    def test_sort_index_intervals(self):
        s = Series([np.nan, 1, 2, 3], IntervalIndex.from_arrays([0, 1, 2, 3], [1, 2, 3, 4]))
        result = s.sort_index()
        expected = s
        tm.assert_series_equal(result, expected)
        result = s.sort_index(ascending=False)
        expected = Series([3, 2, 1, np.nan], IntervalIndex.from_arrays([3, 2, 1, 0], [4, 3, 2, 1]))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('inplace', [True, False])
    @pytest.mark.parametrize('original_list, sorted_list, ascending, ignore_index, output_index', [([2, 3, 6, 1], [2, 3, 6, 1], True, True, [0, 1, 2, 3]), ([2, 3, 6, 1], [2, 3, 6, 1], True, False, [0, 1, 2, 3]), ([2, 3, 6, 1], [1, 6, 3, 2], False, True, [0, 1, 2, 3]), ([2, 3, 6, 1], [1, 6, 3, 2], False, False, [3, 2, 1, 0])])
    def test_sort_index_ignore_index(self, inplace, original_list, sorted_list, ascending, ignore_index, output_index):
        ser = Series(original_list)
        expected = Series(sorted_list, index=output_index)
        kwargs = {'ascending': ascending, 'ignore_index': ignore_index, 'inplace': inplace}
        if inplace:
            result_ser = ser.copy()
            result_ser.sort_index(**kwargs)
        else:
            result_ser = ser.sort_index(**kwargs)
        tm.assert_series_equal(result_ser, expected)
        tm.assert_series_equal(ser, Series(original_list))

    def test_sort_index_ascending_list(self):
        arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'], ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two'], [4, 3, 2, 1, 4, 3, 2, 1]]
        tuples = zip(*arrays)
        mi = MultiIndex.from_tuples(tuples, names=['first', 'second', 'third'])
        ser = Series(range(8), index=mi)
        result = ser.sort_index(level=['third', 'first'], ascending=False)
        expected = ser.iloc[[4, 0, 5, 1, 6, 2, 7, 3]]
        tm.assert_series_equal(result, expected)
        result = ser.sort_index(level=['third', 'first'], ascending=[False, True])
        expected = ser.iloc[[0, 4, 1, 5, 2, 6, 3, 7]]
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('ascending', [None, (True, None), (False, 'True')])
    def test_sort_index_ascending_bad_value_raises(self, ascending):
        ser = Series(range(10), index=[0, 3, 2, 1, 4, 5, 7, 6, 8, 9])
        match = 'For argument "ascending" expected type bool'
        with pytest.raises(ValueError, match=match):
            ser.sort_index(ascending=ascending)