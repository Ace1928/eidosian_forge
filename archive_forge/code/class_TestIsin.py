from datetime import datetime
import struct
import numpy as np
import pytest
from pandas._libs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
import pandas.core.common as com
class TestIsin:

    def test_invalid(self):
        msg = 'only list-like objects are allowed to be passed to isin\\(\\), you passed a `int`'
        with pytest.raises(TypeError, match=msg):
            algos.isin(1, 1)
        with pytest.raises(TypeError, match=msg):
            algos.isin(1, [1])
        with pytest.raises(TypeError, match=msg):
            algos.isin([1], 1)

    def test_basic(self):
        msg = 'isin with argument that is not not a Series'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = algos.isin([1, 2], [1])
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)
        result = algos.isin(np.array([1, 2]), [1])
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)
        result = algos.isin(Series([1, 2]), [1])
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)
        result = algos.isin(Series([1, 2]), Series([1]))
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)
        result = algos.isin(Series([1, 2]), {1})
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = algos.isin(['a', 'b'], ['a'])
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)
        result = algos.isin(Series(['a', 'b']), Series(['a']))
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)
        result = algos.isin(Series(['a', 'b']), {'a'})
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = algos.isin(['a', 'b'], [1])
        expected = np.array([False, False])
        tm.assert_numpy_array_equal(result, expected)

    def test_i8(self):
        arr = date_range('20130101', periods=3).values
        result = algos.isin(arr, [arr[0]])
        expected = np.array([True, False, False])
        tm.assert_numpy_array_equal(result, expected)
        result = algos.isin(arr, arr[0:2])
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)
        result = algos.isin(arr, set(arr[0:2]))
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)
        arr = timedelta_range('1 day', periods=3).values
        result = algos.isin(arr, [arr[0]])
        expected = np.array([True, False, False])
        tm.assert_numpy_array_equal(result, expected)
        result = algos.isin(arr, arr[0:2])
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)
        result = algos.isin(arr, set(arr[0:2]))
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('dtype1', ['m8[ns]', 'M8[ns]', 'M8[ns, UTC]', 'period[D]'])
    @pytest.mark.parametrize('dtype', ['i8', 'f8', 'u8'])
    def test_isin_datetimelike_values_numeric_comps(self, dtype, dtype1):
        dta = date_range('2013-01-01', periods=3)._values
        arr = Series(dta.view('i8')).array.view(dtype1)
        comps = arr.view('i8').astype(dtype)
        result = algos.isin(comps, arr)
        expected = np.zeros(comps.shape, dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

    def test_large(self):
        s = date_range('20000101', periods=2000000, freq='s').values
        result = algos.isin(s, s[0:2])
        expected = np.zeros(len(s), dtype=bool)
        expected[0] = True
        expected[1] = True
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('dtype', ['m8[ns]', 'M8[ns]', 'M8[ns, UTC]', 'period[D]'])
    def test_isin_datetimelike_all_nat(self, dtype):
        dta = date_range('2013-01-01', periods=3)._values
        arr = Series(dta.view('i8')).array.view(dtype)
        arr[0] = NaT
        result = algos.isin(arr, [NaT])
        expected = np.array([True, False, False], dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('dtype', ['m8[ns]', 'M8[ns]', 'M8[ns, UTC]'])
    def test_isin_datetimelike_strings_deprecated(self, dtype):
        dta = date_range('2013-01-01', periods=3)._values
        arr = Series(dta.view('i8')).array.view(dtype)
        vals = [str(x) for x in arr]
        msg = "The behavior of 'isin' with dtype=.* is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            res = algos.isin(arr, vals)
        assert res.all()
        vals2 = np.array(vals, dtype=str)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            res2 = algos.isin(arr, vals2)
        assert res2.all()

    def test_isin_dt64tz_with_nat(self):
        dti = date_range('2016-01-01', periods=3, tz='UTC')
        ser = Series(dti)
        ser[0] = NaT
        res = algos.isin(ser._values, [NaT])
        exp = np.array([True, False, False], dtype=bool)
        tm.assert_numpy_array_equal(res, exp)

    def test_categorical_from_codes(self):
        vals = np.array([0, 1, 2, 0])
        cats = ['a', 'b', 'c']
        Sd = Series(Categorical([1]).from_codes(vals, cats))
        St = Series(Categorical([1]).from_codes(np.array([0, 1]), cats))
        expected = np.array([True, True, False, True])
        result = algos.isin(Sd, St)
        tm.assert_numpy_array_equal(expected, result)

    def test_categorical_isin(self):
        vals = np.array([0, 1, 2, 0])
        cats = ['a', 'b', 'c']
        cat = Categorical([1]).from_codes(vals, cats)
        other = Categorical([1]).from_codes(np.array([0, 1]), cats)
        expected = np.array([True, True, False, True])
        result = algos.isin(cat, other)
        tm.assert_numpy_array_equal(expected, result)

    def test_same_nan_is_in(self):
        comps = [np.nan]
        values = [np.nan]
        expected = np.array([True])
        msg = 'isin with argument that is not not a Series'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = algos.isin(comps, values)
        tm.assert_numpy_array_equal(expected, result)

    def test_same_nan_is_in_large(self):
        s = np.tile(1.0, 1000001)
        s[0] = np.nan
        result = algos.isin(s, np.array([np.nan, 1]))
        expected = np.ones(len(s), dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

    def test_same_nan_is_in_large_series(self):
        s = np.tile(1.0, 1000001)
        series = Series(s)
        s[0] = np.nan
        result = series.isin(np.array([np.nan, 1]))
        expected = Series(np.ones(len(s), dtype=bool))
        tm.assert_series_equal(result, expected)

    def test_same_object_is_in(self):

        class LikeNan:

            def __eq__(self, other) -> bool:
                return False

            def __hash__(self):
                return 0
        a, b = (LikeNan(), LikeNan())
        msg = 'isin with argument that is not not a Series'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            tm.assert_numpy_array_equal(algos.isin([a], [a]), np.array([True]))
            tm.assert_numpy_array_equal(algos.isin([a], [b]), np.array([False]))

    def test_different_nans(self):
        comps = [float('nan')]
        values = [float('nan')]
        assert comps[0] is not values[0]
        result = algos.isin(np.array(comps), values)
        tm.assert_numpy_array_equal(np.array([True]), result)
        result = algos.isin(np.asarray(comps, dtype=object), np.asarray(values, dtype=object))
        tm.assert_numpy_array_equal(np.array([True]), result)
        result = algos.isin(np.asarray(comps, dtype=np.float64), np.asarray(values, dtype=np.float64))
        tm.assert_numpy_array_equal(np.array([True]), result)

    def test_no_cast(self):
        comps = ['ss', 42]
        values = ['42']
        expected = np.array([False, False])
        msg = 'isin with argument that is not not a Series, Index'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = algos.isin(comps, values)
        tm.assert_numpy_array_equal(expected, result)

    @pytest.mark.parametrize('empty', [[], Series(dtype=object), np.array([])])
    def test_empty(self, empty):
        vals = Index(['a', 'b'])
        expected = np.array([False, False])
        result = algos.isin(vals, empty)
        tm.assert_numpy_array_equal(expected, result)

    def test_different_nan_objects(self):
        comps = np.array(['nan', np.nan * 1j, float('nan')], dtype=object)
        vals = np.array([float('nan')], dtype=object)
        expected = np.array([False, False, True])
        result = algos.isin(comps, vals)
        tm.assert_numpy_array_equal(expected, result)

    def test_different_nans_as_float64(self):
        NAN1 = struct.unpack('d', struct.pack('=Q', 9221120237041090560))[0]
        NAN2 = struct.unpack('d', struct.pack('=Q', 9221120237041090561))[0]
        assert NAN1 != NAN1
        assert NAN2 != NAN2
        arr = np.array([NAN1, NAN2], dtype=np.float64)
        lookup1 = np.array([NAN1], dtype=np.float64)
        result = algos.isin(arr, lookup1)
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)
        lookup2 = np.array([NAN2], dtype=np.float64)
        result = algos.isin(arr, lookup2)
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)

    def test_isin_int_df_string_search(self):
        """Comparing df with int`s (1,2) with a string at isin() ("1")
        -> should not match values because int 1 is not equal str 1"""
        df = DataFrame({'values': [1, 2]})
        result = df.isin(['1'])
        expected_false = DataFrame({'values': [False, False]})
        tm.assert_frame_equal(result, expected_false)

    def test_isin_nan_df_string_search(self):
        """Comparing df with nan value (np.nan,2) with a string at isin() ("NaN")
        -> should not match values because np.nan is not equal str NaN"""
        df = DataFrame({'values': [np.nan, 2]})
        result = df.isin(np.array(['NaN'], dtype=object))
        expected_false = DataFrame({'values': [False, False]})
        tm.assert_frame_equal(result, expected_false)

    def test_isin_float_df_string_search(self):
        """Comparing df with floats (1.4245,2.32441) with a string at isin() ("1.4245")
        -> should not match values because float 1.4245 is not equal str 1.4245"""
        df = DataFrame({'values': [1.4245, 2.32441]})
        result = df.isin(np.array(['1.4245'], dtype=object))
        expected_false = DataFrame({'values': [False, False]})
        tm.assert_frame_equal(result, expected_false)

    def test_isin_unsigned_dtype(self):
        ser = Series([1378774140726870442], dtype=np.uint64)
        result = ser.isin([1378774140726870528])
        expected = Series(False)
        tm.assert_series_equal(result, expected)