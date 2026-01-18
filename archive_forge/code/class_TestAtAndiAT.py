from datetime import (
import itertools
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
class TestAtAndiAT:

    def test_float_index_at_iat(self):
        ser = Series([1, 2, 3], index=[0.1, 0.2, 0.3])
        for el, item in ser.items():
            assert ser.at[el] == item
        for i in range(len(ser)):
            assert ser.iat[i] == i + 1

    def test_at_iat_coercion(self):
        dates = date_range('1/1/2000', periods=8)
        df = DataFrame(np.random.default_rng(2).standard_normal((8, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
        s = df['A']
        result = s.at[dates[5]]
        xp = s.values[5]
        assert result == xp

    @pytest.mark.parametrize('ser, expected', [[Series(['2014-01-01', '2014-02-02'], dtype='datetime64[ns]'), Timestamp('2014-02-02')], [Series(['1 days', '2 days'], dtype='timedelta64[ns]'), Timedelta('2 days')]])
    def test_iloc_iat_coercion_datelike(self, indexer_ial, ser, expected):
        result = indexer_ial(ser)[1]
        assert result == expected

    def test_imethods_with_dups(self):
        s = Series(range(5), index=[1, 1, 2, 2, 3], dtype='int64')
        result = s.iloc[2]
        assert result == 2
        result = s.iat[2]
        assert result == 2
        msg = 'index 10 is out of bounds for axis 0 with size 5'
        with pytest.raises(IndexError, match=msg):
            s.iat[10]
        msg = 'index -10 is out of bounds for axis 0 with size 5'
        with pytest.raises(IndexError, match=msg):
            s.iat[-10]
        result = s.iloc[[2, 3]]
        expected = Series([2, 3], [2, 2], dtype='int64')
        tm.assert_series_equal(result, expected)
        df = s.to_frame()
        result = df.iloc[2]
        expected = Series(2, index=[0], name=2)
        tm.assert_series_equal(result, expected)
        result = df.iat[2, 0]
        assert result == 2

    def test_frame_at_with_duplicate_axes(self):
        arr = np.random.default_rng(2).standard_normal(6).reshape(3, 2)
        df = DataFrame(arr, columns=['A', 'A'])
        result = df.at[0, 'A']
        expected = df.iloc[0].copy()
        tm.assert_series_equal(result, expected)
        result = df.T.at['A', 0]
        tm.assert_series_equal(result, expected)
        df.at[1, 'A'] = 2
        expected = Series([2.0, 2.0], index=['A', 'A'], name=1)
        tm.assert_series_equal(df.iloc[1], expected)

    def test_at_getitem_dt64tz_values(self):
        df = DataFrame({'name': ['John', 'Anderson'], 'date': [Timestamp(2017, 3, 13, 13, 32, 56), Timestamp(2017, 2, 16, 12, 10, 3)]})
        df['date'] = df['date'].dt.tz_localize('Asia/Shanghai')
        expected = Timestamp('2017-03-13 13:32:56+0800', tz='Asia/Shanghai')
        result = df.loc[0, 'date']
        assert result == expected
        result = df.at[0, 'date']
        assert result == expected

    def test_mixed_index_at_iat_loc_iloc_series(self):
        s = Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 1, 2])
        for el, item in s.items():
            assert s.at[el] == s.loc[el] == item
        for i in range(len(s)):
            assert s.iat[i] == s.iloc[i] == i + 1
        with pytest.raises(KeyError, match='^4$'):
            s.at[4]
        with pytest.raises(KeyError, match='^4$'):
            s.loc[4]

    def test_mixed_index_at_iat_loc_iloc_dataframe(self):
        df = DataFrame([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], columns=['a', 'b', 'c', 1, 2])
        for rowIdx, row in df.iterrows():
            for el, item in row.items():
                assert df.at[rowIdx, el] == df.loc[rowIdx, el] == item
        for row in range(2):
            for i in range(5):
                assert df.iat[row, i] == df.iloc[row, i] == row * 5 + i
        with pytest.raises(KeyError, match='^3$'):
            df.at[0, 3]
        with pytest.raises(KeyError, match='^3$'):
            df.loc[0, 3]

    def test_iat_setter_incompatible_assignment(self):
        result = DataFrame({'a': [0.0, 1.0], 'b': [4, 5]})
        result.iat[0, 0] = None
        expected = DataFrame({'a': [None, 1], 'b': [4, 5]})
        tm.assert_frame_equal(result, expected)