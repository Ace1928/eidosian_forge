from datetime import (
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
class TestAtSetItem:

    def test_at_setitem_item_cache_cleared(self):
        df = DataFrame(index=[0])
        df['x'] = 1
        df['cost'] = 2
        df['cost']
        df.loc[[0]]
        df.at[0, 'x'] = 4
        df.at[0, 'cost'] = 789
        expected = DataFrame({'x': [4], 'cost': 789}, index=[0], columns=Index(['x', 'cost'], dtype=object))
        tm.assert_frame_equal(df, expected)
        tm.assert_series_equal(df['cost'], expected['cost'])

    def test_at_setitem_mixed_index_assignment(self):
        ser = Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 1, 2])
        ser.at['a'] = 11
        assert ser.iat[0] == 11
        ser.at[1] = 22
        assert ser.iat[3] == 22

    def test_at_setitem_categorical_missing(self):
        df = DataFrame(index=range(3), columns=range(3), dtype=CategoricalDtype(['foo', 'bar']))
        df.at[1, 1] = 'foo'
        expected = DataFrame([[np.nan, np.nan, np.nan], [np.nan, 'foo', np.nan], [np.nan, np.nan, np.nan]], dtype=CategoricalDtype(['foo', 'bar']))
        tm.assert_frame_equal(df, expected)

    def test_at_setitem_multiindex(self):
        df = DataFrame(np.zeros((3, 2), dtype='int64'), columns=MultiIndex.from_tuples([('a', 0), ('a', 1)]))
        df.at[0, 'a'] = 10
        expected = DataFrame([[10, 10], [0, 0], [0, 0]], columns=MultiIndex.from_tuples([('a', 0), ('a', 1)]))
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('row', (Timestamp('2019-01-01'), '2019-01-01'))
    def test_at_datetime_index(self, row):
        df = DataFrame(data=[[1] * 2], index=DatetimeIndex(data=['2019-01-01', '2019-01-02'])).astype({0: 'float64'})
        expected = DataFrame(data=[[0.5, 1], [1.0, 1]], index=DatetimeIndex(data=['2019-01-01', '2019-01-02']))
        df.at[row, 0] = 0.5
        tm.assert_frame_equal(df, expected)