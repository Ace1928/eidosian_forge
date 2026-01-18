import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestDataFrameDiff:

    def test_diff_requires_integer(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((2, 2)))
        with pytest.raises(ValueError, match='periods must be an integer'):
            df.diff(1.5)

    @pytest.mark.parametrize('num', [1, np.int64(1)])
    def test_diff(self, datetime_frame, num):
        df = datetime_frame
        the_diff = df.diff(num)
        expected = df['A'] - df['A'].shift(num)
        tm.assert_series_equal(the_diff['A'], expected)

    def test_diff_int_dtype(self):
        a = 10000000000000000
        b = a + 1
        ser = Series([a, b])
        rs = DataFrame({'s': ser}).diff()
        assert rs.s[1] == 1

    def test_diff_mixed_numeric(self, datetime_frame):
        tf = datetime_frame.astype('float32')
        the_diff = tf.diff(1)
        tm.assert_series_equal(the_diff['A'], tf['A'] - tf['A'].shift(1))

    def test_diff_axis1_nonconsolidated(self):
        df = DataFrame({'y': Series([2]), 'z': Series([3])})
        df.insert(0, 'x', 1)
        result = df.diff(axis=1)
        expected = DataFrame({'x': np.nan, 'y': Series(1), 'z': Series(1)})
        tm.assert_frame_equal(result, expected)

    def test_diff_timedelta64_with_nat(self):
        arr = np.arange(6).reshape(3, 2).astype('timedelta64[ns]')
        arr[:, 0] = np.timedelta64('NaT', 'ns')
        df = DataFrame(arr)
        result = df.diff(1, axis=0)
        expected = DataFrame({0: df[0], 1: [pd.NaT, pd.Timedelta(2), pd.Timedelta(2)]})
        tm.assert_equal(result, expected)
        result = df.diff(0)
        expected = df - df
        assert expected[0].isna().all()
        tm.assert_equal(result, expected)
        result = df.diff(-1, axis=1)
        expected = df * np.nan
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('tz', [None, 'UTC'])
    def test_diff_datetime_axis0_with_nat(self, tz, unit):
        dti = pd.DatetimeIndex(['NaT', '2019-01-01', '2019-01-02'], tz=tz).as_unit(unit)
        ser = Series(dti)
        df = ser.to_frame()
        result = df.diff()
        ex_index = pd.TimedeltaIndex([pd.NaT, pd.NaT, pd.Timedelta(days=1)]).as_unit(unit)
        expected = Series(ex_index).to_frame()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('tz', [None, 'UTC'])
    def test_diff_datetime_with_nat_zero_periods(self, tz):
        dti = date_range('2016-01-01', periods=4, tz=tz)
        ser = Series(dti)
        df = ser.to_frame().copy()
        df[1] = ser.copy()
        df.iloc[:, 0] = pd.NaT
        expected = df - df
        assert expected[0].isna().all()
        result = df.diff(0, axis=0)
        tm.assert_frame_equal(result, expected)
        result = df.diff(0, axis=1)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('tz', [None, 'UTC'])
    def test_diff_datetime_axis0(self, tz):
        df = DataFrame({0: date_range('2010', freq='D', periods=2, tz=tz), 1: date_range('2010', freq='D', periods=2, tz=tz)})
        result = df.diff(axis=0)
        expected = DataFrame({0: pd.TimedeltaIndex(['NaT', '1 days']), 1: pd.TimedeltaIndex(['NaT', '1 days'])})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('tz', [None, 'UTC'])
    def test_diff_datetime_axis1(self, tz):
        df = DataFrame({0: date_range('2010', freq='D', periods=2, tz=tz), 1: date_range('2010', freq='D', periods=2, tz=tz)})
        result = df.diff(axis=1)
        expected = DataFrame({0: pd.TimedeltaIndex(['NaT', 'NaT']), 1: pd.TimedeltaIndex(['0 days', '0 days'])})
        tm.assert_frame_equal(result, expected)

    def test_diff_timedelta(self, unit):
        df = DataFrame({'time': [Timestamp('20130101 9:01'), Timestamp('20130101 9:02')], 'value': [1.0, 2.0]})
        df['time'] = df['time'].dt.as_unit(unit)
        res = df.diff()
        exp = DataFrame([[pd.NaT, np.nan], [pd.Timedelta('00:01:00'), 1]], columns=['time', 'value'])
        exp['time'] = exp['time'].dt.as_unit(unit)
        tm.assert_frame_equal(res, exp)

    def test_diff_mixed_dtype(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        df['A'] = np.array([1, 2, 3, 4, 5], dtype=object)
        result = df.diff()
        assert result[0].dtype == np.float64

    def test_diff_neg_n(self, datetime_frame):
        rs = datetime_frame.diff(-1)
        xp = datetime_frame - datetime_frame.shift(-1)
        tm.assert_frame_equal(rs, xp)

    def test_diff_float_n(self, datetime_frame):
        rs = datetime_frame.diff(1.0)
        xp = datetime_frame.diff(1)
        tm.assert_frame_equal(rs, xp)

    def test_diff_axis(self):
        df = DataFrame([[1.0, 2.0], [3.0, 4.0]])
        tm.assert_frame_equal(df.diff(axis=1), DataFrame([[np.nan, 1.0], [np.nan, 1.0]]))
        tm.assert_frame_equal(df.diff(axis=0), DataFrame([[np.nan, np.nan], [2.0, 2.0]]))

    def test_diff_period(self):
        pi = date_range('2016-01-01', periods=3).to_period('D')
        df = DataFrame({'A': pi})
        result = df.diff(1, axis=1)
        expected = (df - pd.NaT).astype(object)
        tm.assert_frame_equal(result, expected)

    def test_diff_axis1_mixed_dtypes(self):
        df = DataFrame({'A': range(3), 'B': 2 * np.arange(3, dtype=np.float64)})
        expected = DataFrame({'A': [np.nan, np.nan, np.nan], 'B': df['B'] / 2})
        result = df.diff(axis=1)
        tm.assert_frame_equal(result, expected)
        df = DataFrame({'a': np.arange(3, dtype='float32'), 'b': np.arange(3, dtype='float64')})
        result = df.diff(axis=1)
        expected = DataFrame({'a': df['a'] * np.nan, 'b': df['b'] * 0})
        tm.assert_frame_equal(result, expected)

    def test_diff_axis1_mixed_dtypes_large_periods(self):
        df = DataFrame({'A': range(3), 'B': 2 * np.arange(3, dtype=np.float64)})
        expected = df * np.nan
        result = df.diff(axis=1, periods=3)
        tm.assert_frame_equal(result, expected)

    def test_diff_axis1_mixed_dtypes_negative_periods(self):
        df = DataFrame({'A': range(3), 'B': 2 * np.arange(3, dtype=np.float64)})
        expected = DataFrame({'A': -1.0 * df['A'], 'B': df['B'] * np.nan})
        result = df.diff(axis=1, periods=-1)
        tm.assert_frame_equal(result, expected)

    def test_diff_sparse(self):
        sparse_df = DataFrame([[0, 1], [1, 0]], dtype='Sparse[int]')
        result = sparse_df.diff()
        expected = DataFrame([[np.nan, np.nan], [1.0, -1.0]], dtype=pd.SparseDtype('float', 0.0))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('axis,expected', [(0, DataFrame({'a': [np.nan, 0, 1, 0, np.nan, np.nan, np.nan, 0], 'b': [np.nan, 1, np.nan, np.nan, -2, 1, np.nan, np.nan], 'c': np.repeat(np.nan, 8), 'd': [np.nan, 3, 5, 7, 9, 11, 13, 15]}, dtype='Int64')), (1, DataFrame({'a': np.repeat(np.nan, 8), 'b': [0, 1, np.nan, 1, np.nan, np.nan, np.nan, 0], 'c': np.repeat(np.nan, 8), 'd': np.repeat(np.nan, 8)}, dtype='Int64'))])
    def test_diff_integer_na(self, axis, expected):
        df = DataFrame({'a': np.repeat([0, 1, np.nan, 2], 2), 'b': np.tile([0, 1, np.nan, 2], 2), 'c': np.repeat(np.nan, 8), 'd': np.arange(1, 9) ** 2}, dtype='Int64')
        result = df.diff(axis=axis)
        tm.assert_frame_equal(result, expected)

    def test_diff_readonly(self):
        arr = np.random.default_rng(2).standard_normal((5, 2))
        arr.flags.writeable = False
        df = DataFrame(arr)
        result = df.diff()
        expected = DataFrame(np.array(df)).diff()
        tm.assert_frame_equal(result, expected)

    def test_diff_all_int_dtype(self, any_int_numpy_dtype):
        df = DataFrame(range(5))
        df = df.astype(any_int_numpy_dtype)
        result = df.diff()
        expected_dtype = 'float32' if any_int_numpy_dtype in ('int8', 'int16') else 'float64'
        expected = DataFrame([np.nan, 1.0, 1.0, 1.0, 1.0], dtype=expected_dtype)
        tm.assert_frame_equal(result, expected)