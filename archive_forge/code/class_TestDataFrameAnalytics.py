from datetime import timedelta
from decimal import Decimal
import re
from dateutil.tz import tzlocal
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import (
class TestDataFrameAnalytics:

    @pytest.mark.parametrize('axis', [0, 1])
    @pytest.mark.parametrize('opname', ['count', 'sum', 'mean', 'product', 'median', 'min', 'max', 'nunique', 'var', 'std', 'sem', pytest.param('skew', marks=td.skip_if_no('scipy')), pytest.param('kurt', marks=td.skip_if_no('scipy'))])
    def test_stat_op_api_float_string_frame(self, float_string_frame, axis, opname, using_infer_string):
        if (opname in ('sum', 'min', 'max') and axis == 0 or opname in ('count', 'nunique')) and (not (using_infer_string and opname == 'sum')):
            getattr(float_string_frame, opname)(axis=axis)
        else:
            if opname in ['var', 'std', 'sem', 'skew', 'kurt']:
                msg = "could not convert string to float: 'bar'"
            elif opname == 'product':
                if axis == 1:
                    msg = "can't multiply sequence by non-int of type 'float'"
                else:
                    msg = "can't multiply sequence by non-int of type 'str'"
            elif opname == 'sum':
                msg = "unsupported operand type\\(s\\) for \\+: 'float' and 'str'"
            elif opname == 'mean':
                if axis == 0:
                    msg = '|'.join(["Could not convert \\['.*'\\] to numeric", "Could not convert string '(bar){30}' to numeric"])
                else:
                    msg = "unsupported operand type\\(s\\) for \\+: 'float' and 'str'"
            elif opname in ['min', 'max']:
                msg = "'[><]=' not supported between instances of 'float' and 'str'"
            elif opname == 'median':
                msg = re.compile('Cannot convert \\[.*\\] to numeric|does not support', flags=re.S)
            if not isinstance(msg, re.Pattern):
                msg = msg + '|does not support'
            with pytest.raises(TypeError, match=msg):
                getattr(float_string_frame, opname)(axis=axis)
        if opname != 'nunique':
            getattr(float_string_frame, opname)(axis=axis, numeric_only=True)

    @pytest.mark.parametrize('axis', [0, 1])
    @pytest.mark.parametrize('opname', ['count', 'sum', 'mean', 'product', 'median', 'min', 'max', 'var', 'std', 'sem', pytest.param('skew', marks=td.skip_if_no('scipy')), pytest.param('kurt', marks=td.skip_if_no('scipy'))])
    def test_stat_op_api_float_frame(self, float_frame, axis, opname):
        getattr(float_frame, opname)(axis=axis, numeric_only=False)

    def test_stat_op_calc(self, float_frame_with_na, mixed_float_frame):

        def count(s):
            return notna(s).sum()

        def nunique(s):
            return len(algorithms.unique1d(s.dropna()))

        def var(x):
            return np.var(x, ddof=1)

        def std(x):
            return np.std(x, ddof=1)

        def sem(x):
            return np.std(x, ddof=1) / np.sqrt(len(x))
        assert_stat_op_calc('nunique', nunique, float_frame_with_na, has_skipna=False, check_dtype=False, check_dates=True)
        assert_stat_op_calc('sum', np.sum, mixed_float_frame.astype('float32'), check_dtype=False, rtol=0.001)
        assert_stat_op_calc('sum', np.sum, float_frame_with_na, skipna_alternative=np.nansum)
        assert_stat_op_calc('mean', np.mean, float_frame_with_na, check_dates=True)
        assert_stat_op_calc('product', np.prod, float_frame_with_na, skipna_alternative=np.nanprod)
        assert_stat_op_calc('var', var, float_frame_with_na)
        assert_stat_op_calc('std', std, float_frame_with_na)
        assert_stat_op_calc('sem', sem, float_frame_with_na)
        assert_stat_op_calc('count', count, float_frame_with_na, has_skipna=False, check_dtype=False, check_dates=True)

    def test_stat_op_calc_skew_kurtosis(self, float_frame_with_na):
        sp_stats = pytest.importorskip('scipy.stats')

        def skewness(x):
            if len(x) < 3:
                return np.nan
            return sp_stats.skew(x, bias=False)

        def kurt(x):
            if len(x) < 4:
                return np.nan
            return sp_stats.kurtosis(x, bias=False)
        assert_stat_op_calc('skew', skewness, float_frame_with_na)
        assert_stat_op_calc('kurt', kurt, float_frame_with_na)

    def test_median(self, float_frame_with_na, int_frame):

        def wrapper(x):
            if isna(x).any():
                return np.nan
            return np.median(x)
        assert_stat_op_calc('median', wrapper, float_frame_with_na, check_dates=True)
        assert_stat_op_calc('median', wrapper, int_frame, check_dtype=False, check_dates=True)

    @pytest.mark.parametrize('method', ['sum', 'mean', 'prod', 'var', 'std', 'skew', 'min', 'max'])
    @pytest.mark.parametrize('df', [DataFrame({'a': [-0.0004998754019959134, -0.001646725777291983, 0.0006769587077588301], 'b': [-0, -0, 0.0], 'c': [0.00031111847529610595, 0.0014902627951905339, -0.0009409920003597969]}, index=['foo', 'bar', 'baz'], dtype='O'), DataFrame({0: [np.nan, 2], 1: [np.nan, 3], 2: [np.nan, 4]}, dtype=object)])
    @pytest.mark.filterwarnings('ignore:Mismatched null-like values:FutureWarning')
    def test_stat_operators_attempt_obj_array(self, method, df, axis):
        assert df.values.dtype == np.object_
        result = getattr(df, method)(axis=axis)
        expected = getattr(df.astype('f8'), method)(axis=axis).astype(object)
        if axis in [1, 'columns'] and method in ['min', 'max']:
            expected[expected.isna()] = None
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('op', ['mean', 'std', 'var', 'skew', 'kurt', 'sem'])
    def test_mixed_ops(self, op):
        df = DataFrame({'int': [1, 2, 3, 4], 'float': [1.0, 2.0, 3.0, 4.0], 'str': ['a', 'b', 'c', 'd']})
        msg = '|'.join(['Could not convert', 'could not convert', "can't multiply sequence by non-int", 'does not support'])
        with pytest.raises(TypeError, match=msg):
            getattr(df, op)()
        with pd.option_context('use_bottleneck', False):
            msg = '|'.join(['Could not convert', 'could not convert', "can't multiply sequence by non-int", 'does not support'])
            with pytest.raises(TypeError, match=msg):
                getattr(df, op)()

    @pytest.mark.xfail(using_pyarrow_string_dtype(), reason="sum doesn't work for arrow strings")
    def test_reduce_mixed_frame(self):
        df = DataFrame({'bool_data': [True, True, False, False, False], 'int_data': [10, 20, 30, 40, 50], 'string_data': ['a', 'b', 'c', 'd', 'e']})
        df.reindex(columns=['bool_data', 'int_data', 'string_data'])
        test = df.sum(axis=0)
        tm.assert_numpy_array_equal(test.values, np.array([2, 150, 'abcde'], dtype=object))
        alt = df.T.sum(axis=1)
        tm.assert_series_equal(test, alt)

    def test_nunique(self):
        df = DataFrame({'A': [1, 1, 1], 'B': [1, 2, 3], 'C': [1, np.nan, 3]})
        tm.assert_series_equal(df.nunique(), Series({'A': 1, 'B': 3, 'C': 2}))
        tm.assert_series_equal(df.nunique(dropna=False), Series({'A': 1, 'B': 3, 'C': 3}))
        tm.assert_series_equal(df.nunique(axis=1), Series({0: 1, 1: 2, 2: 2}))
        tm.assert_series_equal(df.nunique(axis=1, dropna=False), Series({0: 1, 1: 3, 2: 2}))

    @pytest.mark.parametrize('tz', [None, 'UTC'])
    def test_mean_mixed_datetime_numeric(self, tz):
        df = DataFrame({'A': [1, 1], 'B': [Timestamp('2000', tz=tz)] * 2})
        result = df.mean()
        expected = Series([1.0, Timestamp('2000', tz=tz)], index=['A', 'B'])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('tz', [None, 'UTC'])
    def test_mean_includes_datetimes(self, tz):
        df = DataFrame({'A': [Timestamp('2000', tz=tz)] * 2})
        result = df.mean()
        expected = Series([Timestamp('2000', tz=tz)], index=['A'])
        tm.assert_series_equal(result, expected)

    def test_mean_mixed_string_decimal(self):
        d = [{'A': 2, 'B': None, 'C': Decimal('628.00')}, {'A': 1, 'B': None, 'C': Decimal('383.00')}, {'A': 3, 'B': None, 'C': Decimal('651.00')}, {'A': 2, 'B': None, 'C': Decimal('575.00')}, {'A': 4, 'B': None, 'C': Decimal('1114.00')}, {'A': 1, 'B': 'TEST', 'C': Decimal('241.00')}, {'A': 2, 'B': None, 'C': Decimal('572.00')}, {'A': 4, 'B': None, 'C': Decimal('609.00')}, {'A': 3, 'B': None, 'C': Decimal('820.00')}, {'A': 5, 'B': None, 'C': Decimal('1223.00')}]
        df = DataFrame(d)
        with pytest.raises(TypeError, match='unsupported operand type|does not support'):
            df.mean()
        result = df[['A', 'C']].mean()
        expected = Series([2.7, 681.6], index=['A', 'C'], dtype=object)
        tm.assert_series_equal(result, expected)

    def test_var_std(self, datetime_frame):
        result = datetime_frame.std(ddof=4)
        expected = datetime_frame.apply(lambda x: x.std(ddof=4))
        tm.assert_almost_equal(result, expected)
        result = datetime_frame.var(ddof=4)
        expected = datetime_frame.apply(lambda x: x.var(ddof=4))
        tm.assert_almost_equal(result, expected)
        arr = np.repeat(np.random.default_rng(2).random((1, 1000)), 1000, 0)
        result = nanops.nanvar(arr, axis=0)
        assert not (result < 0).any()
        with pd.option_context('use_bottleneck', False):
            result = nanops.nanvar(arr, axis=0)
            assert not (result < 0).any()

    @pytest.mark.parametrize('meth', ['sem', 'var', 'std'])
    def test_numeric_only_flag(self, meth):
        df1 = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), columns=['foo', 'bar', 'baz'])
        df1 = df1.astype({'foo': object})
        df1.loc[0, 'foo'] = '100'
        df2 = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), columns=['foo', 'bar', 'baz'])
        df2 = df2.astype({'foo': object})
        df2.loc[0, 'foo'] = 'a'
        result = getattr(df1, meth)(axis=1, numeric_only=True)
        expected = getattr(df1[['bar', 'baz']], meth)(axis=1)
        tm.assert_series_equal(expected, result)
        result = getattr(df2, meth)(axis=1, numeric_only=True)
        expected = getattr(df2[['bar', 'baz']], meth)(axis=1)
        tm.assert_series_equal(expected, result)
        msg = "unsupported operand type\\(s\\) for -: 'float' and 'str'"
        with pytest.raises(TypeError, match=msg):
            getattr(df1, meth)(axis=1, numeric_only=False)
        msg = "could not convert string to float: 'a'"
        with pytest.raises(TypeError, match=msg):
            getattr(df2, meth)(axis=1, numeric_only=False)

    def test_sem(self, datetime_frame):
        result = datetime_frame.sem(ddof=4)
        expected = datetime_frame.apply(lambda x: x.std(ddof=4) / np.sqrt(len(x)))
        tm.assert_almost_equal(result, expected)
        arr = np.repeat(np.random.default_rng(2).random((1, 1000)), 1000, 0)
        result = nanops.nansem(arr, axis=0)
        assert not (result < 0).any()
        with pd.option_context('use_bottleneck', False):
            result = nanops.nansem(arr, axis=0)
            assert not (result < 0).any()

    @pytest.mark.parametrize('dropna, expected', [(True, {'A': [12], 'B': [10.0], 'C': [1.0], 'D': ['a'], 'E': Categorical(['a'], categories=['a']), 'F': DatetimeIndex(['2000-01-02'], dtype='M8[ns]'), 'G': to_timedelta(['1 days'])}), (False, {'A': [12], 'B': [10.0], 'C': [np.nan], 'D': np.array([np.nan], dtype=object), 'E': Categorical([np.nan], categories=['a']), 'F': DatetimeIndex([pd.NaT], dtype='M8[ns]'), 'G': to_timedelta([pd.NaT])}), (True, {'H': [8, 9, np.nan, np.nan], 'I': [8, 9, np.nan, np.nan], 'J': [1, np.nan, np.nan, np.nan], 'K': Categorical(['a', np.nan, np.nan, np.nan], categories=['a']), 'L': DatetimeIndex(['2000-01-02', 'NaT', 'NaT', 'NaT'], dtype='M8[ns]'), 'M': to_timedelta(['1 days', 'nan', 'nan', 'nan']), 'N': [0, 1, 2, 3]}), (False, {'H': [8, 9, np.nan, np.nan], 'I': [8, 9, np.nan, np.nan], 'J': [1, np.nan, np.nan, np.nan], 'K': Categorical([np.nan, 'a', np.nan, np.nan], categories=['a']), 'L': DatetimeIndex(['NaT', '2000-01-02', 'NaT', 'NaT'], dtype='M8[ns]'), 'M': to_timedelta(['nan', '1 days', 'nan', 'nan']), 'N': [0, 1, 2, 3]})])
    def test_mode_dropna(self, dropna, expected):
        df = DataFrame({'A': [12, 12, 19, 11], 'B': [10, 10, np.nan, 3], 'C': [1, np.nan, np.nan, np.nan], 'D': Series([np.nan, np.nan, 'a', np.nan], dtype=object), 'E': Categorical([np.nan, np.nan, 'a', np.nan]), 'F': DatetimeIndex(['NaT', '2000-01-02', 'NaT', 'NaT'], dtype='M8[ns]'), 'G': to_timedelta(['1 days', 'nan', 'nan', 'nan']), 'H': [8, 8, 9, 9], 'I': [9, 9, 8, 8], 'J': [1, 1, np.nan, np.nan], 'K': Categorical(['a', np.nan, 'a', np.nan]), 'L': DatetimeIndex(['2000-01-02', '2000-01-02', 'NaT', 'NaT'], dtype='M8[ns]'), 'M': to_timedelta(['1 days', 'nan', '1 days', 'nan']), 'N': np.arange(4, dtype='int64')})
        result = df[sorted(expected.keys())].mode(dropna=dropna)
        expected = DataFrame(expected)
        tm.assert_frame_equal(result, expected)

    def test_mode_sortwarning(self, using_infer_string):
        df = DataFrame({'A': [np.nan, np.nan, 'a', 'a']})
        expected = DataFrame({'A': ['a', np.nan]})
        warning = None if using_infer_string else UserWarning
        with tm.assert_produces_warning(warning):
            result = df.mode(dropna=False)
            result = result.sort_values(by='A').reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

    def test_mode_empty_df(self):
        df = DataFrame([], columns=['a', 'b'])
        result = df.mode()
        expected = DataFrame([], columns=['a', 'b'], index=Index([], dtype=np.int64))
        tm.assert_frame_equal(result, expected)

    def test_operators_timedelta64(self):
        df = DataFrame({'A': date_range('2012-1-1', periods=3, freq='D'), 'B': date_range('2012-1-2', periods=3, freq='D'), 'C': Timestamp('20120101') - timedelta(minutes=5, seconds=5)})
        diffs = DataFrame({'A': df['A'] - df['C'], 'B': df['A'] - df['B']})
        result = diffs.min()
        assert result.iloc[0] == diffs.loc[0, 'A']
        assert result.iloc[1] == diffs.loc[0, 'B']
        result = diffs.min(axis=1)
        assert (result == diffs.loc[0, 'B']).all()
        result = diffs.max()
        assert result.iloc[0] == diffs.loc[2, 'A']
        assert result.iloc[1] == diffs.loc[2, 'B']
        result = diffs.max(axis=1)
        assert (result == diffs['A']).all()
        result = diffs.abs()
        result2 = abs(diffs)
        expected = DataFrame({'A': df['A'] - df['C'], 'B': df['B'] - df['A']})
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)
        mixed = diffs.copy()
        mixed['C'] = 'foo'
        mixed['D'] = 1
        mixed['E'] = 1.0
        mixed['F'] = Timestamp('20130101')
        result = mixed.min()
        expected = Series([pd.Timedelta(timedelta(seconds=5 * 60 + 5)), pd.Timedelta(timedelta(days=-1)), 'foo', 1, 1.0, Timestamp('20130101')], index=mixed.columns)
        tm.assert_series_equal(result, expected)
        result = mixed.min(axis=1, numeric_only=True)
        expected = Series([1, 1, 1.0], index=[0, 1, 2])
        tm.assert_series_equal(result, expected)
        result = mixed[['A', 'B']].min(1)
        expected = Series([timedelta(days=-1)] * 3)
        tm.assert_series_equal(result, expected)
        result = mixed[['A', 'B']].min()
        expected = Series([timedelta(seconds=5 * 60 + 5), timedelta(days=-1)], index=['A', 'B'])
        tm.assert_series_equal(result, expected)
        df = DataFrame({'time': date_range('20130102', periods=5), 'time2': date_range('20130105', periods=5)})
        df['off1'] = df['time2'] - df['time']
        assert df['off1'].dtype == 'timedelta64[ns]'
        df['off2'] = df['time'] - df['time2']
        df._consolidate_inplace()
        assert df['off1'].dtype == 'timedelta64[ns]'
        assert df['off2'].dtype == 'timedelta64[ns]'

    def test_std_timedelta64_skipna_false(self):
        tdi = pd.timedelta_range('1 Day', periods=10)
        df = DataFrame({'A': tdi, 'B': tdi}, copy=True)
        df.iloc[-2, -1] = pd.NaT
        result = df.std(skipna=False)
        expected = Series([df['A'].std(), pd.NaT], index=['A', 'B'], dtype='timedelta64[ns]')
        tm.assert_series_equal(result, expected)
        result = df.std(axis=1, skipna=False)
        expected = Series([pd.Timedelta(0)] * 8 + [pd.NaT, pd.Timedelta(0)])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('values', [['2022-01-01', '2022-01-02', pd.NaT, '2022-01-03'], 4 * [pd.NaT]])
    def test_std_datetime64_with_nat(self, values, skipna, using_array_manager, request, unit):
        if using_array_manager and (not skipna or all((value is pd.NaT for value in values))):
            mark = pytest.mark.xfail(reason='GH#51446: Incorrect type inference on NaT in reduction result')
            request.applymarker(mark)
        dti = to_datetime(values).as_unit(unit)
        df = DataFrame({'a': dti})
        result = df.std(skipna=skipna)
        if not skipna or all((value is pd.NaT for value in values)):
            expected = Series({'a': pd.NaT}, dtype=f'timedelta64[{unit}]')
        else:
            expected = Series({'a': 86400000000000}, dtype=f'timedelta64[{unit}]')
        tm.assert_series_equal(result, expected)

    def test_sum_corner(self):
        empty_frame = DataFrame()
        axis0 = empty_frame.sum(0)
        axis1 = empty_frame.sum(1)
        assert isinstance(axis0, Series)
        assert isinstance(axis1, Series)
        assert len(axis0) == 0
        assert len(axis1) == 0

    @pytest.mark.parametrize('index', [RangeIndex(0), DatetimeIndex([]), Index([], dtype=np.int64), Index([], dtype=np.float64), DatetimeIndex([], freq='ME'), PeriodIndex([], freq='D')])
    def test_axis_1_empty(self, all_reductions, index):
        df = DataFrame(columns=['a'], index=index)
        result = getattr(df, all_reductions)(axis=1)
        if all_reductions in ('any', 'all'):
            expected_dtype = 'bool'
        elif all_reductions == 'count':
            expected_dtype = 'int64'
        else:
            expected_dtype = 'object'
        expected = Series([], index=index, dtype=expected_dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('method, unit', [('sum', 0), ('prod', 1)])
    @pytest.mark.parametrize('numeric_only', [None, True, False])
    def test_sum_prod_nanops(self, method, unit, numeric_only):
        idx = ['a', 'b', 'c']
        df = DataFrame({'a': [unit, unit], 'b': [unit, np.nan], 'c': [np.nan, np.nan]})
        result = getattr(df, method)(numeric_only=numeric_only)
        expected = Series([unit, unit, unit], index=idx, dtype='float64')
        tm.assert_series_equal(result, expected)
        result = getattr(df, method)(numeric_only=numeric_only, min_count=1)
        expected = Series([unit, unit, np.nan], index=idx)
        tm.assert_series_equal(result, expected)
        result = getattr(df, method)(numeric_only=numeric_only, min_count=0)
        expected = Series([unit, unit, unit], index=idx, dtype='float64')
        tm.assert_series_equal(result, expected)
        result = getattr(df.iloc[1:], method)(numeric_only=numeric_only, min_count=1)
        expected = Series([unit, np.nan, np.nan], index=idx)
        tm.assert_series_equal(result, expected)
        df = DataFrame({'A': [unit] * 10, 'B': [unit] * 5 + [np.nan] * 5})
        result = getattr(df, method)(numeric_only=numeric_only, min_count=5)
        expected = Series(result, index=['A', 'B'])
        tm.assert_series_equal(result, expected)
        result = getattr(df, method)(numeric_only=numeric_only, min_count=6)
        expected = Series(result, index=['A', 'B'])
        tm.assert_series_equal(result, expected)

    def test_sum_nanops_timedelta(self):
        idx = ['a', 'b', 'c']
        df = DataFrame({'a': [0, 0], 'b': [0, np.nan], 'c': [np.nan, np.nan]})
        df2 = df.apply(to_timedelta)
        result = df2.sum()
        expected = Series([0, 0, 0], dtype='m8[ns]', index=idx)
        tm.assert_series_equal(result, expected)
        result = df2.sum(min_count=0)
        tm.assert_series_equal(result, expected)
        result = df2.sum(min_count=1)
        expected = Series([0, 0, np.nan], dtype='m8[ns]', index=idx)
        tm.assert_series_equal(result, expected)

    def test_sum_nanops_min_count(self):
        df = DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        result = df.sum(min_count=10)
        expected = Series([np.nan, np.nan], index=['x', 'y'])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('float_type', ['float16', 'float32', 'float64'])
    @pytest.mark.parametrize('kwargs, expected_result', [({'axis': 1, 'min_count': 2}, [3.2, 5.3, np.nan]), ({'axis': 1, 'min_count': 3}, [np.nan, np.nan, np.nan]), ({'axis': 1, 'skipna': False}, [3.2, 5.3, np.nan])])
    def test_sum_nanops_dtype_min_count(self, float_type, kwargs, expected_result):
        df = DataFrame({'a': [1.0, 2.3, 4.4], 'b': [2.2, 3, np.nan]}, dtype=float_type)
        result = df.sum(**kwargs)
        expected = Series(expected_result).astype(float_type)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('float_type', ['float16', 'float32', 'float64'])
    @pytest.mark.parametrize('kwargs, expected_result', [({'axis': 1, 'min_count': 2}, [2.0, 4.0, np.nan]), ({'axis': 1, 'min_count': 3}, [np.nan, np.nan, np.nan]), ({'axis': 1, 'skipna': False}, [2.0, 4.0, np.nan])])
    def test_prod_nanops_dtype_min_count(self, float_type, kwargs, expected_result):
        df = DataFrame({'a': [1.0, 2.0, 4.4], 'b': [2.0, 2.0, np.nan]}, dtype=float_type)
        result = df.prod(**kwargs)
        expected = Series(expected_result).astype(float_type)
        tm.assert_series_equal(result, expected)

    def test_sum_object(self, float_frame):
        values = float_frame.values.astype(int)
        frame = DataFrame(values, index=float_frame.index, columns=float_frame.columns)
        deltas = frame * timedelta(1)
        deltas.sum()

    def test_sum_bool(self, float_frame):
        bools = np.isnan(float_frame)
        bools.sum(1)
        bools.sum(0)

    def test_sum_mixed_datetime(self):
        df = DataFrame({'A': date_range('2000', periods=4), 'B': [1, 2, 3, 4]}).reindex([2, 3, 4])
        with pytest.raises(TypeError, match="does not support reduction 'sum'"):
            df.sum()

    def test_mean_corner(self, float_frame, float_string_frame):
        msg = 'Could not convert|does not support'
        with pytest.raises(TypeError, match=msg):
            float_string_frame.mean(axis=0)
        with pytest.raises(TypeError, match='unsupported operand type'):
            float_string_frame.mean(axis=1)
        float_frame['bool'] = float_frame['A'] > 0
        means = float_frame.mean(0)
        assert means['bool'] == float_frame['bool'].values.mean()

    def test_mean_datetimelike(self):
        df = DataFrame({'A': np.arange(3), 'B': date_range('2016-01-01', periods=3), 'C': pd.timedelta_range('1D', periods=3), 'D': pd.period_range('2016', periods=3, freq='Y')})
        result = df.mean(numeric_only=True)
        expected = Series({'A': 1.0})
        tm.assert_series_equal(result, expected)
        with pytest.raises(TypeError, match='mean is not implemented for PeriodArray'):
            df.mean()

    def test_mean_datetimelike_numeric_only_false(self):
        df = DataFrame({'A': np.arange(3), 'B': date_range('2016-01-01', periods=3), 'C': pd.timedelta_range('1D', periods=3)})
        result = df.mean(numeric_only=False)
        expected = Series({'A': 1, 'B': df.loc[1, 'B'], 'C': df.loc[1, 'C']})
        tm.assert_series_equal(result, expected)
        df['D'] = pd.period_range('2016', periods=3, freq='Y')
        with pytest.raises(TypeError, match='mean is not implemented for Period'):
            df.mean(numeric_only=False)

    def test_mean_extensionarray_numeric_only_true(self):
        arr = np.random.default_rng(2).integers(1000, size=(10, 5))
        df = DataFrame(arr, dtype='Int64')
        result = df.mean(numeric_only=True)
        expected = DataFrame(arr).mean().astype('Float64')
        tm.assert_series_equal(result, expected)

    def test_stats_mixed_type(self, float_string_frame):
        with pytest.raises(TypeError, match='could not convert'):
            float_string_frame.std(1)
        with pytest.raises(TypeError, match='could not convert'):
            float_string_frame.var(1)
        with pytest.raises(TypeError, match='unsupported operand type'):
            float_string_frame.mean(1)
        with pytest.raises(TypeError, match='could not convert'):
            float_string_frame.skew(1)

    def test_sum_bools(self):
        df = DataFrame(index=range(1), columns=range(10))
        bools = isna(df)
        assert bools.sum(axis=1)[0] == 10

    @pytest.mark.parametrize('skipna', [True, False])
    @pytest.mark.parametrize('axis', [0, 1])
    def test_idxmin(self, float_frame, int_frame, skipna, axis):
        frame = float_frame
        frame.iloc[5:10] = np.nan
        frame.iloc[15:20, -2:] = np.nan
        for df in [frame, int_frame]:
            warn = None
            if skipna is False or axis == 1:
                warn = None if df is int_frame else FutureWarning
            msg = 'The behavior of DataFrame.idxmin with all-NA values'
            with tm.assert_produces_warning(warn, match=msg):
                result = df.idxmin(axis=axis, skipna=skipna)
            msg2 = 'The behavior of Series.idxmin'
            with tm.assert_produces_warning(warn, match=msg2):
                expected = df.apply(Series.idxmin, axis=axis, skipna=skipna)
            expected = expected.astype(df.index.dtype)
            tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('axis', [0, 1])
    @pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
    def test_idxmin_empty(self, index, skipna, axis):
        if axis == 0:
            frame = DataFrame(index=index)
        else:
            frame = DataFrame(columns=index)
        result = frame.idxmin(axis=axis, skipna=skipna)
        expected = Series(dtype=index.dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('numeric_only', [True, False])
    def test_idxmin_numeric_only(self, numeric_only):
        df = DataFrame({'a': [2, 3, 1], 'b': [2, 1, 1], 'c': list('xyx')})
        result = df.idxmin(numeric_only=numeric_only)
        if numeric_only:
            expected = Series([2, 1], index=['a', 'b'])
        else:
            expected = Series([2, 1, 0], index=['a', 'b', 'c'])
        tm.assert_series_equal(result, expected)

    def test_idxmin_axis_2(self, float_frame):
        frame = float_frame
        msg = 'No axis named 2 for object type DataFrame'
        with pytest.raises(ValueError, match=msg):
            frame.idxmin(axis=2)

    @pytest.mark.parametrize('skipna', [True, False])
    @pytest.mark.parametrize('axis', [0, 1])
    def test_idxmax(self, float_frame, int_frame, skipna, axis):
        frame = float_frame
        frame.iloc[5:10] = np.nan
        frame.iloc[15:20, -2:] = np.nan
        for df in [frame, int_frame]:
            warn = None
            if skipna is False or axis == 1:
                warn = None if df is int_frame else FutureWarning
            msg = 'The behavior of DataFrame.idxmax with all-NA values'
            with tm.assert_produces_warning(warn, match=msg):
                result = df.idxmax(axis=axis, skipna=skipna)
            msg2 = 'The behavior of Series.idxmax'
            with tm.assert_produces_warning(warn, match=msg2):
                expected = df.apply(Series.idxmax, axis=axis, skipna=skipna)
            expected = expected.astype(df.index.dtype)
            tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('axis', [0, 1])
    @pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
    def test_idxmax_empty(self, index, skipna, axis):
        if axis == 0:
            frame = DataFrame(index=index)
        else:
            frame = DataFrame(columns=index)
        result = frame.idxmax(axis=axis, skipna=skipna)
        expected = Series(dtype=index.dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('numeric_only', [True, False])
    def test_idxmax_numeric_only(self, numeric_only):
        df = DataFrame({'a': [2, 3, 1], 'b': [2, 1, 1], 'c': list('xyx')})
        result = df.idxmax(numeric_only=numeric_only)
        if numeric_only:
            expected = Series([1, 0], index=['a', 'b'])
        else:
            expected = Series([1, 0, 1], index=['a', 'b', 'c'])
        tm.assert_series_equal(result, expected)

    def test_idxmax_arrow_types(self):
        pytest.importorskip('pyarrow')
        df = DataFrame({'a': [2, 3, 1], 'b': [2, 1, 1]}, dtype='int64[pyarrow]')
        result = df.idxmax()
        expected = Series([1, 0], index=['a', 'b'])
        tm.assert_series_equal(result, expected)
        result = df.idxmin()
        expected = Series([2, 1], index=['a', 'b'])
        tm.assert_series_equal(result, expected)
        df = DataFrame({'a': ['b', 'c', 'a']}, dtype='string[pyarrow]')
        result = df.idxmax(numeric_only=False)
        expected = Series([1], index=['a'])
        tm.assert_series_equal(result, expected)
        result = df.idxmin(numeric_only=False)
        expected = Series([2], index=['a'])
        tm.assert_series_equal(result, expected)

    def test_idxmax_axis_2(self, float_frame):
        frame = float_frame
        msg = 'No axis named 2 for object type DataFrame'
        with pytest.raises(ValueError, match=msg):
            frame.idxmax(axis=2)

    def test_idxmax_mixed_dtype(self):
        dti = date_range('2016-01-01', periods=3)
        df = DataFrame({1: [0, 2, 1], 2: range(3)[::-1], 3: dti.copy(deep=True)})
        result = df.idxmax()
        expected = Series([1, 0, 2], index=[1, 2, 3])
        tm.assert_series_equal(result, expected)
        result = df.idxmin()
        expected = Series([0, 2, 0], index=[1, 2, 3])
        tm.assert_series_equal(result, expected)
        df.loc[0, 3] = pd.NaT
        result = df.idxmax()
        expected = Series([1, 0, 2], index=[1, 2, 3])
        tm.assert_series_equal(result, expected)
        result = df.idxmin()
        expected = Series([0, 2, 1], index=[1, 2, 3])
        tm.assert_series_equal(result, expected)
        df[4] = dti[::-1]
        df._consolidate_inplace()
        result = df.idxmax()
        expected = Series([1, 0, 2, 0], index=[1, 2, 3, 4])
        tm.assert_series_equal(result, expected)
        result = df.idxmin()
        expected = Series([0, 2, 1, 2], index=[1, 2, 3, 4])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('op, expected_value', [('idxmax', [0, 4]), ('idxmin', [0, 5])])
    def test_idxmax_idxmin_convert_dtypes(self, op, expected_value):
        df = DataFrame({'ID': [100, 100, 100, 200, 200, 200], 'value': [0, 0, 0, 1, 2, 0]}, dtype='Int64')
        df = df.groupby('ID')
        result = getattr(df, op)()
        expected = DataFrame({'value': expected_value}, index=Index([100, 200], name='ID', dtype='Int64'))
        tm.assert_frame_equal(result, expected)

    def test_idxmax_dt64_multicolumn_axis1(self):
        dti = date_range('2016-01-01', periods=3)
        df = DataFrame({3: dti, 4: dti[::-1]}, copy=True)
        df.iloc[0, 0] = pd.NaT
        df._consolidate_inplace()
        result = df.idxmax(axis=1)
        expected = Series([4, 3, 3])
        tm.assert_series_equal(result, expected)
        result = df.idxmin(axis=1)
        expected = Series([4, 3, 4])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('opname', ['any', 'all'])
    @pytest.mark.parametrize('axis', [0, 1])
    @pytest.mark.parametrize('bool_only', [False, True])
    def test_any_all_mixed_float(self, opname, axis, bool_only, float_string_frame):
        mixed = float_string_frame
        mixed['_bool_'] = np.random.default_rng(2).standard_normal(len(mixed)) > 0.5
        getattr(mixed, opname)(axis=axis, bool_only=bool_only)

    @pytest.mark.parametrize('opname', ['any', 'all'])
    @pytest.mark.parametrize('axis', [0, 1])
    def test_any_all_bool_with_na(self, opname, axis, bool_frame_with_na):
        getattr(bool_frame_with_na, opname)(axis=axis, bool_only=False)

    @pytest.mark.filterwarnings('ignore:Downcasting object dtype arrays:FutureWarning')
    @pytest.mark.parametrize('opname', ['any', 'all'])
    def test_any_all_bool_frame(self, opname, bool_frame_with_na):
        frame = bool_frame_with_na.fillna(True)
        alternative = getattr(np, opname)
        f = getattr(frame, opname)

        def skipna_wrapper(x):
            nona = x.dropna().values
            return alternative(nona)

        def wrapper(x):
            return alternative(x.values)
        result0 = f(axis=0, skipna=False)
        result1 = f(axis=1, skipna=False)
        tm.assert_series_equal(result0, frame.apply(wrapper))
        tm.assert_series_equal(result1, frame.apply(wrapper, axis=1))
        result0 = f(axis=0)
        result1 = f(axis=1)
        tm.assert_series_equal(result0, frame.apply(skipna_wrapper))
        tm.assert_series_equal(result1, frame.apply(skipna_wrapper, axis=1), check_dtype=False)
        with pytest.raises(ValueError, match='No axis named 2'):
            f(axis=2)
        all_na = frame * np.nan
        r0 = getattr(all_na, opname)(axis=0)
        r1 = getattr(all_na, opname)(axis=1)
        if opname == 'any':
            assert not r0.any()
            assert not r1.any()
        else:
            assert r0.all()
            assert r1.all()

    def test_any_all_extra(self):
        df = DataFrame({'A': [True, False, False], 'B': [True, True, False], 'C': [True, True, True]}, index=['a', 'b', 'c'])
        result = df[['A', 'B']].any(axis=1)
        expected = Series([True, True, False], index=['a', 'b', 'c'])
        tm.assert_series_equal(result, expected)
        result = df[['A', 'B']].any(axis=1, bool_only=True)
        tm.assert_series_equal(result, expected)
        result = df.all(1)
        expected = Series([True, False, False], index=['a', 'b', 'c'])
        tm.assert_series_equal(result, expected)
        result = df.all(1, bool_only=True)
        tm.assert_series_equal(result, expected)
        result = df.all(axis=None).item()
        assert result is False
        result = df.any(axis=None).item()
        assert result is True
        result = df[['C']].all(axis=None).item()
        assert result is True

    @pytest.mark.parametrize('axis', [0, 1])
    @pytest.mark.parametrize('bool_agg_func', ['any', 'all'])
    @pytest.mark.parametrize('skipna', [True, False])
    def test_any_all_object_dtype(self, axis, bool_agg_func, skipna, using_infer_string):
        df = DataFrame(data=[[1, np.nan, np.nan, True], [np.nan, 2, np.nan, True], [np.nan, np.nan, np.nan, True], [np.nan, np.nan, '5', np.nan]])
        if using_infer_string:
            val = not axis == 0 and (not skipna) and (bool_agg_func == 'all')
        else:
            val = True
        result = getattr(df, bool_agg_func)(axis=axis, skipna=skipna)
        expected = Series([True, True, val, True])
        tm.assert_series_equal(result, expected)

    @pytest.mark.filterwarnings("ignore:'any' with datetime64 dtypes is deprecated.*:FutureWarning")
    def test_any_datetime(self):
        float_data = [1, np.nan, 3, np.nan]
        datetime_data = [Timestamp('1960-02-15'), Timestamp('1960-02-16'), pd.NaT, pd.NaT]
        df = DataFrame({'A': float_data, 'B': datetime_data})
        result = df.any(axis=1)
        expected = Series([True, True, True, False])
        tm.assert_series_equal(result, expected)

    def test_any_all_bool_only(self):
        df = DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': [None, None, None]}, columns=Index(['col1', 'col2', 'col3'], dtype=object))
        result = df.all(bool_only=True)
        expected = Series(dtype=np.bool_, index=[])
        tm.assert_series_equal(result, expected)
        df = DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': [None, None, None], 'col4': [False, False, True]})
        result = df.all(bool_only=True)
        expected = Series({'col4': False})
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('func, data, expected', [(np.any, {}, False), (np.all, {}, True), (np.any, {'A': []}, False), (np.all, {'A': []}, True), (np.any, {'A': [False, False]}, False), (np.all, {'A': [False, False]}, False), (np.any, {'A': [True, False]}, True), (np.all, {'A': [True, False]}, False), (np.any, {'A': [True, True]}, True), (np.all, {'A': [True, True]}, True), (np.any, {'A': [False], 'B': [False]}, False), (np.all, {'A': [False], 'B': [False]}, False), (np.any, {'A': [False, False], 'B': [False, True]}, True), (np.all, {'A': [False, False], 'B': [False, True]}, False), (np.all, {'A': Series([0.0, 1.0], dtype='float')}, False), (np.any, {'A': Series([0.0, 1.0], dtype='float')}, True), (np.all, {'A': Series([0, 1], dtype=int)}, False), (np.any, {'A': Series([0, 1], dtype=int)}, True), pytest.param(np.all, {'A': Series([0, 1], dtype='M8[ns]')}, False), pytest.param(np.all, {'A': Series([0, 1], dtype='M8[ns, UTC]')}, False), pytest.param(np.any, {'A': Series([0, 1], dtype='M8[ns]')}, True), pytest.param(np.any, {'A': Series([0, 1], dtype='M8[ns, UTC]')}, True), pytest.param(np.all, {'A': Series([1, 2], dtype='M8[ns]')}, True), pytest.param(np.all, {'A': Series([1, 2], dtype='M8[ns, UTC]')}, True), pytest.param(np.any, {'A': Series([1, 2], dtype='M8[ns]')}, True), pytest.param(np.any, {'A': Series([1, 2], dtype='M8[ns, UTC]')}, True), pytest.param(np.all, {'A': Series([0, 1], dtype='m8[ns]')}, False), pytest.param(np.any, {'A': Series([0, 1], dtype='m8[ns]')}, True), pytest.param(np.all, {'A': Series([1, 2], dtype='m8[ns]')}, True), pytest.param(np.any, {'A': Series([1, 2], dtype='m8[ns]')}, True), (np.all, {'A': Series([0, 1], dtype='category')}, True), (np.any, {'A': Series([0, 1], dtype='category')}, False), (np.all, {'A': Series([1, 2], dtype='category')}, True), (np.any, {'A': Series([1, 2], dtype='category')}, False), pytest.param(np.all, {'A': Series([10, 20], dtype='M8[ns]'), 'B': Series([10, 20], dtype='m8[ns]')}, True)])
    def test_any_all_np_func(self, func, data, expected):
        data = DataFrame(data)
        if any((isinstance(x, CategoricalDtype) for x in data.dtypes)):
            with pytest.raises(TypeError, match='dtype category does not support reduction'):
                func(data)
            with pytest.raises(TypeError, match='dtype category does not support reduction'):
                getattr(DataFrame(data), func.__name__)(axis=None)
        else:
            msg = "'(any|all)' with datetime64 dtypes is deprecated"
            if data.dtypes.apply(lambda x: x.kind == 'M').any():
                warn = FutureWarning
            else:
                warn = None
            with tm.assert_produces_warning(warn, match=msg, check_stacklevel=False):
                result = func(data)
            assert isinstance(result, np.bool_)
            assert result.item() is expected
            with tm.assert_produces_warning(warn, match=msg):
                result = getattr(DataFrame(data), func.__name__)(axis=None)
            assert isinstance(result, np.bool_)
            assert result.item() is expected

    def test_any_all_object(self):
        result = np.all(DataFrame(columns=['a', 'b'])).item()
        assert result is True
        result = np.any(DataFrame(columns=['a', 'b'])).item()
        assert result is False

    def test_any_all_object_bool_only(self):
        df = DataFrame({'A': ['foo', 2], 'B': [True, False]}).astype(object)
        df._consolidate_inplace()
        df['C'] = Series([True, True])
        df['D'] = df['C'].astype('category')
        res = df._get_bool_data()
        expected = df[['C']]
        tm.assert_frame_equal(res, expected)
        res = df.all(bool_only=True, axis=0)
        expected = Series([True], index=['C'])
        tm.assert_series_equal(res, expected)
        res = df[['B', 'C']].all(bool_only=True, axis=0)
        tm.assert_series_equal(res, expected)
        assert df.all(bool_only=True, axis=None)
        res = df.any(bool_only=True, axis=0)
        expected = Series([True], index=['C'])
        tm.assert_series_equal(res, expected)
        res = df[['C']].any(bool_only=True, axis=0)
        tm.assert_series_equal(res, expected)
        assert df.any(bool_only=True, axis=None)

    def test_series_broadcasting(self):
        df = DataFrame([1.0, 1.0, 1.0])
        df_nan = DataFrame({'A': [np.nan, 2.0, np.nan]})
        s = Series([1, 1, 1])
        s_nan = Series([np.nan, np.nan, 1])
        with tm.assert_produces_warning(None):
            df_nan.clip(lower=s, axis=0)
            for op in ['lt', 'le', 'gt', 'ge', 'eq', 'ne']:
                getattr(df, op)(s_nan, axis=0)