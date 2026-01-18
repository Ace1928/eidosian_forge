import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import algorithms
from pandas.core.arrays import PeriodArray
class TestSeriesIsIn:

    def test_isin(self):
        s = Series(['A', 'B', 'C', 'a', 'B', 'B', 'A', 'C'])
        result = s.isin(['A', 'C'])
        expected = Series([True, False, True, False, False, False, True, True])
        tm.assert_series_equal(result, expected)
        s = Series(list('abcdefghijk' * 10 ** 5))
        in_list = [-1, 'a', 'b', 'G', 'Y', 'Z', 'E', 'K', 'E', 'S', 'I', 'R', 'R'] * 6
        assert s.isin(in_list).sum() == 200000

    def test_isin_with_string_scalar(self):
        s = Series(['A', 'B', 'C', 'a', 'B', 'B', 'A', 'C'])
        msg = 'only list-like objects are allowed to be passed to isin\\(\\), you passed a `str`'
        with pytest.raises(TypeError, match=msg):
            s.isin('a')
        s = Series(['aaa', 'b', 'c'])
        with pytest.raises(TypeError, match=msg):
            s.isin('aaa')

    def test_isin_datetimelike_mismatched_reso(self):
        expected = Series([True, True, False, False, False])
        ser = Series(date_range('jan-01-2013', 'jan-05-2013'))
        day_values = np.asarray(ser[0:2].values).astype('datetime64[D]')
        result = ser.isin(day_values)
        tm.assert_series_equal(result, expected)
        dta = ser[:2]._values.astype('M8[s]')
        result = ser.isin(dta)
        tm.assert_series_equal(result, expected)

    def test_isin_datetimelike_mismatched_reso_list(self):
        expected = Series([True, True, False, False, False])
        ser = Series(date_range('jan-01-2013', 'jan-05-2013'))
        dta = ser[:2]._values.astype('M8[s]')
        result = ser.isin(list(dta))
        tm.assert_series_equal(result, expected)

    def test_isin_with_i8(self):
        expected = Series([True, True, False, False, False])
        expected2 = Series([False, True, False, False, False])
        s = Series(date_range('jan-01-2013', 'jan-05-2013'))
        result = s.isin(s[0:2])
        tm.assert_series_equal(result, expected)
        result = s.isin(s[0:2].values)
        tm.assert_series_equal(result, expected)
        result = s.isin([s[1]])
        tm.assert_series_equal(result, expected2)
        result = s.isin([np.datetime64(s[1])])
        tm.assert_series_equal(result, expected2)
        result = s.isin(set(s[0:2]))
        tm.assert_series_equal(result, expected)
        s = Series(pd.to_timedelta(range(5), unit='d'))
        result = s.isin(s[0:2])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('empty', [[], Series(dtype=object), np.array([])])
    def test_isin_empty(self, empty):
        s = Series(['a', 'b'])
        expected = Series([False, False])
        result = s.isin(empty)
        tm.assert_series_equal(expected, result)

    def test_isin_read_only(self):
        arr = np.array([1, 2, 3])
        arr.setflags(write=False)
        s = Series([1, 2, 3])
        result = s.isin(arr)
        expected = Series([True, True, True])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('dtype', [object, None])
    def test_isin_dt64_values_vs_ints(self, dtype):
        dti = date_range('2013-01-01', '2013-01-05')
        ser = Series(dti)
        comps = np.asarray([1356998400000000000], dtype=dtype)
        res = dti.isin(comps)
        expected = np.array([False] * len(dti), dtype=bool)
        tm.assert_numpy_array_equal(res, expected)
        res = ser.isin(comps)
        tm.assert_series_equal(res, Series(expected))
        res = pd.core.algorithms.isin(ser, comps)
        tm.assert_numpy_array_equal(res, expected)

    def test_isin_tzawareness_mismatch(self):
        dti = date_range('2013-01-01', '2013-01-05')
        ser = Series(dti)
        other = dti.tz_localize('UTC')
        res = dti.isin(other)
        expected = np.array([False] * len(dti), dtype=bool)
        tm.assert_numpy_array_equal(res, expected)
        res = ser.isin(other)
        tm.assert_series_equal(res, Series(expected))
        res = pd.core.algorithms.isin(ser, other)
        tm.assert_numpy_array_equal(res, expected)

    def test_isin_period_freq_mismatch(self):
        dti = date_range('2013-01-01', '2013-01-05')
        pi = dti.to_period('M')
        ser = Series(pi)
        dtype = dti.to_period('Y').dtype
        other = PeriodArray._simple_new(pi.asi8, dtype=dtype)
        res = pi.isin(other)
        expected = np.array([False] * len(pi), dtype=bool)
        tm.assert_numpy_array_equal(res, expected)
        res = ser.isin(other)
        tm.assert_series_equal(res, Series(expected))
        res = pd.core.algorithms.isin(ser, other)
        tm.assert_numpy_array_equal(res, expected)

    @pytest.mark.parametrize('values', [[-9.0, 0.0], [-9, 0]])
    def test_isin_float_in_int_series(self, values):
        ser = Series(values)
        result = ser.isin([-9, -0.5])
        expected = Series([True, False])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('dtype', ['boolean', 'Int64', 'Float64'])
    @pytest.mark.parametrize('data,values,expected', [([0, 1, 0], [1], [False, True, False]), ([0, 1, 0], [1, pd.NA], [False, True, False]), ([0, pd.NA, 0], [1, 0], [True, False, True]), ([0, 1, pd.NA], [1, pd.NA], [False, True, True]), ([0, 1, pd.NA], [1, np.nan], [False, True, False]), ([0, pd.NA, pd.NA], [np.nan, pd.NaT, None], [False, False, False])])
    def test_isin_masked_types(self, dtype, data, values, expected):
        ser = Series(data, dtype=dtype)
        result = ser.isin(values)
        expected = Series(expected, dtype='boolean')
        tm.assert_series_equal(result, expected)