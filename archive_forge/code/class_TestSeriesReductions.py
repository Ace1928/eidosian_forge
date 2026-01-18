from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
class TestSeriesReductions:

    def test_sum_inf(self):
        s = Series(np.random.default_rng(2).standard_normal(10))
        s2 = s.copy()
        s[5:8] = np.inf
        s2[5:8] = np.nan
        assert np.isinf(s.sum())
        arr = np.random.default_rng(2).standard_normal((100, 100)).astype('f4')
        arr[:, 2] = np.inf
        msg = 'use_inf_as_na option is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            with pd.option_context('mode.use_inf_as_na', True):
                tm.assert_almost_equal(s.sum(), s2.sum())
        res = nanops.nansum(arr, axis=1)
        assert np.isinf(res).all()

    @pytest.mark.parametrize('dtype', ['float64', 'Float32', 'Int64', 'boolean', 'object'])
    @pytest.mark.parametrize('use_bottleneck', [True, False])
    @pytest.mark.parametrize('method, unit', [('sum', 0.0), ('prod', 1.0)])
    def test_empty(self, method, unit, use_bottleneck, dtype):
        with pd.option_context('use_bottleneck', use_bottleneck):
            s = Series([], dtype=dtype)
            result = getattr(s, method)()
            assert result == unit
            result = getattr(s, method)(min_count=0)
            assert result == unit
            result = getattr(s, method)(min_count=1)
            assert isna(result)
            result = getattr(s, method)(skipna=True)
            result == unit
            result = getattr(s, method)(skipna=True, min_count=0)
            assert result == unit
            result = getattr(s, method)(skipna=True, min_count=1)
            assert isna(result)
            result = getattr(s, method)(skipna=False, min_count=0)
            assert result == unit
            result = getattr(s, method)(skipna=False, min_count=1)
            assert isna(result)
            s = Series([np.nan], dtype=dtype)
            result = getattr(s, method)()
            assert result == unit
            result = getattr(s, method)(min_count=0)
            assert result == unit
            result = getattr(s, method)(min_count=1)
            assert isna(result)
            result = getattr(s, method)(skipna=True)
            result == unit
            result = getattr(s, method)(skipna=True, min_count=0)
            assert result == unit
            result = getattr(s, method)(skipna=True, min_count=1)
            assert isna(result)
            s = Series([np.nan, 1], dtype=dtype)
            result = getattr(s, method)()
            assert result == 1.0
            result = getattr(s, method)(min_count=0)
            assert result == 1.0
            result = getattr(s, method)(min_count=1)
            assert result == 1.0
            result = getattr(s, method)(skipna=True)
            assert result == 1.0
            result = getattr(s, method)(skipna=True, min_count=0)
            assert result == 1.0
            df = DataFrame(np.empty((10, 0)), dtype=dtype)
            assert (getattr(df, method)(1) == unit).all()
            s = Series([1], dtype=dtype)
            result = getattr(s, method)(min_count=2)
            assert isna(result)
            result = getattr(s, method)(skipna=False, min_count=2)
            assert isna(result)
            s = Series([np.nan], dtype=dtype)
            result = getattr(s, method)(min_count=2)
            assert isna(result)
            s = Series([np.nan, 1], dtype=dtype)
            result = getattr(s, method)(min_count=2)
            assert isna(result)

    @pytest.mark.parametrize('method', ['mean', 'var'])
    @pytest.mark.parametrize('dtype', ['Float64', 'Int64', 'boolean'])
    def test_ops_consistency_on_empty_nullable(self, method, dtype):
        eser = Series([], dtype=dtype)
        result = getattr(eser, method)()
        assert result is pd.NA
        nser = Series([np.nan], dtype=dtype)
        result = getattr(nser, method)()
        assert result is pd.NA

    @pytest.mark.parametrize('method', ['mean', 'median', 'std', 'var'])
    def test_ops_consistency_on_empty(self, method):
        result = getattr(Series(dtype=float), method)()
        assert isna(result)
        tdser = Series([], dtype='m8[ns]')
        if method == 'var':
            msg = '|'.join(["operation 'var' not allowed", 'cannot perform var with type timedelta64\\[ns\\]', "does not support reduction 'var'"])
            with pytest.raises(TypeError, match=msg):
                getattr(tdser, method)()
        else:
            result = getattr(tdser, method)()
            assert result is NaT

    def test_nansum_buglet(self):
        ser = Series([1.0, np.nan], index=[0, 1])
        result = np.nansum(ser)
        tm.assert_almost_equal(result, 1)

    @pytest.mark.parametrize('use_bottleneck', [True, False])
    @pytest.mark.parametrize('dtype', ['int32', 'int64'])
    def test_sum_overflow_int(self, use_bottleneck, dtype):
        with pd.option_context('use_bottleneck', use_bottleneck):
            v = np.arange(5000000, dtype=dtype)
            s = Series(v)
            result = s.sum(skipna=False)
            assert int(result) == v.sum(dtype='int64')
            result = s.min(skipna=False)
            assert int(result) == 0
            result = s.max(skipna=False)
            assert int(result) == v[-1]

    @pytest.mark.parametrize('use_bottleneck', [True, False])
    @pytest.mark.parametrize('dtype', ['float32', 'float64'])
    def test_sum_overflow_float(self, use_bottleneck, dtype):
        with pd.option_context('use_bottleneck', use_bottleneck):
            v = np.arange(5000000, dtype=dtype)
            s = Series(v)
            result = s.sum(skipna=False)
            assert result == v.sum(dtype=dtype)
            result = s.min(skipna=False)
            assert np.allclose(float(result), 0.0)
            result = s.max(skipna=False)
            assert np.allclose(float(result), v[-1])

    def test_mean_masked_overflow(self):
        val = 100000000000000000
        n_elements = 100
        na = np.array([val] * n_elements)
        ser = Series([val] * n_elements, dtype='Int64')
        result_numpy = np.mean(na)
        result_masked = ser.mean()
        assert result_masked - result_numpy == 0
        assert result_masked == 1e+17

    @pytest.mark.parametrize('ddof, exp', [(1, 2.5), (0, 2.0)])
    def test_var_masked_array(self, ddof, exp):
        ser = Series([1, 2, 3, 4, 5], dtype='Int64')
        ser_numpy_dtype = Series([1, 2, 3, 4, 5], dtype='int64')
        result = ser.var(ddof=ddof)
        result_numpy_dtype = ser_numpy_dtype.var(ddof=ddof)
        assert result == result_numpy_dtype
        assert result == exp

    @pytest.mark.parametrize('dtype', ('m8[ns]', 'm8[ns]', 'M8[ns]', 'M8[ns, UTC]'))
    @pytest.mark.parametrize('skipna', [True, False])
    def test_empty_timeseries_reductions_return_nat(self, dtype, skipna):
        assert Series([], dtype=dtype).min(skipna=skipna) is NaT
        assert Series([], dtype=dtype).max(skipna=skipna) is NaT

    def test_numpy_argmin(self):
        data = np.arange(1, 11)
        s = Series(data, index=data)
        result = np.argmin(s)
        expected = np.argmin(data)
        assert result == expected
        result = s.argmin()
        assert result == expected
        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.argmin(s, out=data)

    def test_numpy_argmax(self):
        data = np.arange(1, 11)
        ser = Series(data, index=data)
        result = np.argmax(ser)
        expected = np.argmax(data)
        assert result == expected
        result = ser.argmax()
        assert result == expected
        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.argmax(ser, out=data)

    def test_idxmin_dt64index(self, unit):
        dti = DatetimeIndex(['NaT', '2015-02-08', 'NaT']).as_unit(unit)
        ser = Series([1.0, 2.0, np.nan], index=dti)
        msg = 'The behavior of Series.idxmin with all-NA values'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            res = ser.idxmin(skipna=False)
        assert res is NaT
        msg = 'The behavior of Series.idxmax with all-NA values'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            res = ser.idxmax(skipna=False)
        assert res is NaT
        df = ser.to_frame()
        msg = 'The behavior of DataFrame.idxmin with all-NA values'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            res = df.idxmin(skipna=False)
        assert res.dtype == f'M8[{unit}]'
        assert res.isna().all()
        msg = 'The behavior of DataFrame.idxmax with all-NA values'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            res = df.idxmax(skipna=False)
        assert res.dtype == f'M8[{unit}]'
        assert res.isna().all()

    def test_idxmin(self):
        string_series = Series(range(20), dtype=np.float64, name='series')
        string_series[5:15] = np.nan
        assert string_series[string_series.idxmin()] == string_series.min()
        msg = 'The behavior of Series.idxmin'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert isna(string_series.idxmin(skipna=False))
        nona = string_series.dropna()
        assert nona[nona.idxmin()] == nona.min()
        assert nona.index.values.tolist().index(nona.idxmin()) == nona.values.argmin()
        allna = string_series * np.nan
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert isna(allna.idxmin())
        s = Series(date_range('20130102', periods=6))
        result = s.idxmin()
        assert result == 0
        s[0] = np.nan
        result = s.idxmin()
        assert result == 1

    def test_idxmax(self):
        string_series = Series(range(20), dtype=np.float64, name='series')
        string_series[5:15] = np.nan
        assert string_series[string_series.idxmax()] == string_series.max()
        msg = 'The behavior of Series.idxmax with all-NA values'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert isna(string_series.idxmax(skipna=False))
        nona = string_series.dropna()
        assert nona[nona.idxmax()] == nona.max()
        assert nona.index.values.tolist().index(nona.idxmax()) == nona.values.argmax()
        allna = string_series * np.nan
        msg = 'The behavior of Series.idxmax with all-NA values'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert isna(allna.idxmax())
        s = Series(date_range('20130102', periods=6))
        result = s.idxmax()
        assert result == 5
        s[5] = np.nan
        result = s.idxmax()
        assert result == 4
        s = Series([1, 2, 3], [1.1, 2.1, 3.1])
        result = s.idxmax()
        assert result == 3.1
        result = s.idxmin()
        assert result == 1.1
        s = Series(s.index, s.index)
        result = s.idxmax()
        assert result == 3.1
        result = s.idxmin()
        assert result == 1.1

    def test_all_any(self):
        ts = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10), name='ts')
        bool_series = ts > 0
        assert not bool_series.all()
        assert bool_series.any()
        s = Series(['abc', True])
        assert s.any()

    def test_numpy_all_any(self, index_or_series):
        idx = index_or_series([0, 1, 2])
        assert not np.all(idx)
        assert np.any(idx)
        idx = Index([1, 2, 3])
        assert np.all(idx)

    def test_all_any_skipna(self):
        s1 = Series([np.nan, True])
        s2 = Series([np.nan, False])
        assert s1.all(skipna=False)
        assert s1.all(skipna=True)
        assert s2.any(skipna=False)
        assert not s2.any(skipna=True)

    def test_all_any_bool_only(self):
        s = Series([False, False, True, True, False, True], index=[0, 0, 1, 1, 2, 2])
        assert s.any(bool_only=True)
        assert not s.all(bool_only=True)

    @pytest.mark.parametrize('bool_agg_func', ['any', 'all'])
    @pytest.mark.parametrize('skipna', [True, False])
    def test_any_all_object_dtype(self, bool_agg_func, skipna):
        ser = Series(['a', 'b', 'c', 'd', 'e'], dtype=object)
        result = getattr(ser, bool_agg_func)(skipna=skipna)
        expected = True
        assert result == expected

    @pytest.mark.parametrize('bool_agg_func', ['any', 'all'])
    @pytest.mark.parametrize('data', [[False, None], [None, False], [False, np.nan], [np.nan, False]])
    def test_any_all_object_dtype_missing(self, data, bool_agg_func):
        ser = Series(data)
        result = getattr(ser, bool_agg_func)(skipna=False)
        expected = bool_agg_func == 'any' and None not in data
        assert result == expected

    @pytest.mark.parametrize('dtype', ['boolean', 'Int64', 'UInt64', 'Float64'])
    @pytest.mark.parametrize('bool_agg_func', ['any', 'all'])
    @pytest.mark.parametrize('skipna', [True, False])
    @pytest.mark.parametrize('data,expected_data', [([0, 0, 0], [[False, False], [False, False]]), ([1, 1, 1], [[True, True], [True, True]]), ([pd.NA, pd.NA, pd.NA], [[pd.NA, pd.NA], [False, True]]), ([0, pd.NA, 0], [[pd.NA, False], [False, False]]), ([1, pd.NA, 1], [[True, pd.NA], [True, True]]), ([1, pd.NA, 0], [[True, False], [True, False]])])
    def test_any_all_nullable_kleene_logic(self, bool_agg_func, skipna, data, dtype, expected_data):
        ser = Series(data, dtype=dtype)
        expected = expected_data[skipna][bool_agg_func == 'all']
        result = getattr(ser, bool_agg_func)(skipna=skipna)
        assert result is pd.NA and expected is pd.NA or result == expected

    def test_any_axis1_bool_only(self):
        df = DataFrame({'A': [True, False], 'B': [1, 2]})
        result = df.any(axis=1, bool_only=True)
        expected = Series([True, False])
        tm.assert_series_equal(result, expected)

    def test_any_all_datetimelike(self):
        dta = date_range('1995-01-02', periods=3)._data
        ser = Series(dta)
        df = DataFrame(ser)
        msg = "'(any|all)' with datetime64 dtypes is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert dta.all()
            assert dta.any()
            assert ser.all()
            assert ser.any()
            assert df.any().all()
            assert df.all().all()
        dta = dta.tz_localize('UTC')
        ser = Series(dta)
        df = DataFrame(ser)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert dta.all()
            assert dta.any()
            assert ser.all()
            assert ser.any()
            assert df.any().all()
            assert df.all().all()
        tda = dta - dta[0]
        ser = Series(tda)
        df = DataFrame(ser)
        assert tda.any()
        assert not tda.all()
        assert ser.any()
        assert not ser.all()
        assert df.any().all()
        assert not df.all().any()

    def test_any_all_pyarrow_string(self):
        pytest.importorskip('pyarrow')
        ser = Series(['', 'a'], dtype='string[pyarrow_numpy]')
        assert ser.any()
        assert not ser.all()
        ser = Series([None, 'a'], dtype='string[pyarrow_numpy]')
        assert ser.any()
        assert ser.all()
        assert not ser.all(skipna=False)
        ser = Series([None, ''], dtype='string[pyarrow_numpy]')
        assert not ser.any()
        assert not ser.all()
        ser = Series(['a', 'b'], dtype='string[pyarrow_numpy]')
        assert ser.any()
        assert ser.all()

    def test_timedelta64_analytics(self):
        dti = date_range('2012-1-1', periods=3, freq='D')
        td = Series(dti) - Timestamp('20120101')
        result = td.idxmin()
        assert result == 0
        result = td.idxmax()
        assert result == 2
        td[0] = np.nan
        result = td.idxmin()
        assert result == 1
        result = td.idxmax()
        assert result == 2
        s1 = Series(date_range('20120101', periods=3))
        s2 = Series(date_range('20120102', periods=3))
        expected = Series(s2 - s1)
        result = np.abs(s1 - s2)
        tm.assert_series_equal(result, expected)
        result = (s1 - s2).abs()
        tm.assert_series_equal(result, expected)
        result = td.max()
        expected = Timedelta('2 days')
        assert result == expected
        result = td.min()
        expected = Timedelta('1 days')
        assert result == expected

    @pytest.mark.parametrize('test_input,error_type', [(Series([], dtype='float64'), ValueError), (Series(['foo', 'bar', 'baz']), TypeError), (Series([(1,), (2,)]), TypeError), (Series(['foo', 'foo', 'bar', 'bar', None, np.nan, 'baz']), TypeError)])
    def test_assert_idxminmax_empty_raises(self, test_input, error_type):
        """
        Cases where ``Series.argmax`` and related should raise an exception
        """
        test_input = Series([], dtype='float64')
        msg = 'attempt to get argmin of an empty sequence'
        with pytest.raises(ValueError, match=msg):
            test_input.idxmin()
        with pytest.raises(ValueError, match=msg):
            test_input.idxmin(skipna=False)
        msg = 'attempt to get argmax of an empty sequence'
        with pytest.raises(ValueError, match=msg):
            test_input.idxmax()
        with pytest.raises(ValueError, match=msg):
            test_input.idxmax(skipna=False)

    def test_idxminmax_object_dtype(self, using_infer_string):
        ser = Series(['foo', 'bar', 'baz'])
        assert ser.idxmax() == 0
        assert ser.idxmax(skipna=False) == 0
        assert ser.idxmin() == 1
        assert ser.idxmin(skipna=False) == 1
        ser2 = Series([(1,), (2,)])
        assert ser2.idxmax() == 1
        assert ser2.idxmax(skipna=False) == 1
        assert ser2.idxmin() == 0
        assert ser2.idxmin(skipna=False) == 0
        if not using_infer_string:
            ser3 = Series(['foo', 'foo', 'bar', 'bar', None, np.nan, 'baz'])
            msg = "'>' not supported between instances of 'float' and 'str'"
            with pytest.raises(TypeError, match=msg):
                ser3.idxmax()
            with pytest.raises(TypeError, match=msg):
                ser3.idxmax(skipna=False)
            msg = "'<' not supported between instances of 'float' and 'str'"
            with pytest.raises(TypeError, match=msg):
                ser3.idxmin()
            with pytest.raises(TypeError, match=msg):
                ser3.idxmin(skipna=False)

    def test_idxminmax_object_frame(self):
        df = DataFrame([['zimm', 2.5], ['biff', 1.0], ['bid', 12.0]])
        res = df.idxmax()
        exp = Series([0, 2])
        tm.assert_series_equal(res, exp)

    def test_idxminmax_object_tuples(self):
        ser = Series([(1, 3), (2, 2), (3, 1)])
        assert ser.idxmax() == 2
        assert ser.idxmin() == 0
        assert ser.idxmax(skipna=False) == 2
        assert ser.idxmin(skipna=False) == 0

    def test_idxminmax_object_decimals(self):
        df = DataFrame({'idx': [0, 1], 'x': [Decimal('8.68'), Decimal('42.23')], 'y': [Decimal('7.11'), Decimal('79.61')]})
        res = df.idxmax()
        exp = Series({'idx': 1, 'x': 1, 'y': 1})
        tm.assert_series_equal(res, exp)
        res2 = df.idxmin()
        exp2 = exp - 1
        tm.assert_series_equal(res2, exp2)

    def test_argminmax_object_ints(self):
        ser = Series([0, 1], dtype='object')
        assert ser.argmax() == 1
        assert ser.argmin() == 0
        assert ser.argmax(skipna=False) == 1
        assert ser.argmin(skipna=False) == 0

    def test_idxminmax_with_inf(self):
        s = Series([0, -np.inf, np.inf, np.nan])
        assert s.idxmin() == 1
        msg = 'The behavior of Series.idxmin with all-NA values'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert np.isnan(s.idxmin(skipna=False))
        assert s.idxmax() == 2
        msg = 'The behavior of Series.idxmax with all-NA values'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert np.isnan(s.idxmax(skipna=False))
        msg = 'use_inf_as_na option is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            with pd.option_context('mode.use_inf_as_na', True):
                assert s.idxmin() == 0
                assert np.isnan(s.idxmin(skipna=False))
                assert s.idxmax() == 0
                np.isnan(s.idxmax(skipna=False))

    def test_sum_uint64(self):
        s = Series([10000000000000000000], dtype='uint64')
        result = s.sum()
        expected = np.uint64(10000000000000000000)
        tm.assert_almost_equal(result, expected)