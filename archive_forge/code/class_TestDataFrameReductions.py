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
class TestDataFrameReductions:

    def test_min_max_dt64_with_NaT(self):
        df = DataFrame({'foo': [pd.NaT, pd.NaT, Timestamp('2012-05-01')]})
        res = df.min()
        exp = Series([Timestamp('2012-05-01')], index=['foo'])
        tm.assert_series_equal(res, exp)
        res = df.max()
        exp = Series([Timestamp('2012-05-01')], index=['foo'])
        tm.assert_series_equal(res, exp)
        df = DataFrame({'foo': [pd.NaT, pd.NaT]})
        res = df.min()
        exp = Series([pd.NaT], index=['foo'])
        tm.assert_series_equal(res, exp)
        res = df.max()
        exp = Series([pd.NaT], index=['foo'])
        tm.assert_series_equal(res, exp)

    def test_min_max_dt64_with_NaT_skipna_false(self, request, tz_naive_fixture):
        tz = tz_naive_fixture
        if isinstance(tz, tzlocal) and is_platform_windows():
            pytest.skip('GH#37659 OSError raised within tzlocal bc Windows chokes in times before 1970-01-01')
        df = DataFrame({'a': [Timestamp('2020-01-01 08:00:00', tz=tz), Timestamp('1920-02-01 09:00:00', tz=tz)], 'b': [Timestamp('2020-02-01 08:00:00', tz=tz), pd.NaT]})
        res = df.min(axis=1, skipna=False)
        expected = Series([df.loc[0, 'a'], pd.NaT])
        assert expected.dtype == df['a'].dtype
        tm.assert_series_equal(res, expected)
        res = df.max(axis=1, skipna=False)
        expected = Series([df.loc[0, 'b'], pd.NaT])
        assert expected.dtype == df['a'].dtype
        tm.assert_series_equal(res, expected)

    def test_min_max_dt64_api_consistency_with_NaT(self):
        df = DataFrame({'x': to_datetime([])})
        expected_dt_series = Series(to_datetime([]))
        assert (df.min(axis=0).x is pd.NaT) == (expected_dt_series.min() is pd.NaT)
        assert (df.max(axis=0).x is pd.NaT) == (expected_dt_series.max() is pd.NaT)
        tm.assert_series_equal(df.min(axis=1), expected_dt_series)
        tm.assert_series_equal(df.max(axis=1), expected_dt_series)

    def test_min_max_dt64_api_consistency_empty_df(self):
        df = DataFrame({'x': []})
        expected_float_series = Series([], dtype=float)
        assert np.isnan(df.min(axis=0).x) == np.isnan(expected_float_series.min())
        assert np.isnan(df.max(axis=0).x) == np.isnan(expected_float_series.max())
        tm.assert_series_equal(df.min(axis=1), expected_float_series)
        tm.assert_series_equal(df.min(axis=1), expected_float_series)

    @pytest.mark.parametrize('initial', ['2018-10-08 13:36:45+00:00', '2018-10-08 13:36:45+03:00'])
    @pytest.mark.parametrize('method', ['min', 'max'])
    def test_preserve_timezone(self, initial: str, method):
        initial_dt = to_datetime(initial)
        expected = Series([initial_dt])
        df = DataFrame([expected])
        result = getattr(df, method)(axis=1)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('method', ['min', 'max'])
    def test_minmax_tzaware_skipna_axis_1(self, method, skipna):
        val = to_datetime('1900-01-01', utc=True)
        df = DataFrame({'a': Series([pd.NaT, pd.NaT, val]), 'b': Series([pd.NaT, val, val])})
        op = getattr(df, method)
        result = op(axis=1, skipna=skipna)
        if skipna:
            expected = Series([pd.NaT, val, val])
        else:
            expected = Series([pd.NaT, pd.NaT, val])
        tm.assert_series_equal(result, expected)

    def test_frame_any_with_timedelta(self):
        df = DataFrame({'a': Series([0, 0]), 't': Series([to_timedelta(0, 's'), to_timedelta(1, 'ms')])})
        result = df.any(axis=0)
        expected = Series(data=[False, True], index=['a', 't'])
        tm.assert_series_equal(result, expected)
        result = df.any(axis=1)
        expected = Series(data=[False, True])
        tm.assert_series_equal(result, expected)

    def test_reductions_skipna_none_raises(self, request, frame_or_series, all_reductions):
        if all_reductions == 'count':
            request.applymarker(pytest.mark.xfail(reason='Count does not accept skipna'))
        obj = frame_or_series([1, 2, 3])
        msg = 'For argument "skipna" expected type bool, received type NoneType.'
        with pytest.raises(ValueError, match=msg):
            getattr(obj, all_reductions)(skipna=None)

    @td.skip_array_manager_invalid_test
    def test_reduction_timestamp_smallest_unit(self):
        df = DataFrame({'a': Series([Timestamp('2019-12-31')], dtype='datetime64[s]'), 'b': Series([Timestamp('2019-12-31 00:00:00.123')], dtype='datetime64[ms]')})
        result = df.max()
        expected = Series([Timestamp('2019-12-31'), Timestamp('2019-12-31 00:00:00.123')], dtype='datetime64[ms]', index=['a', 'b'])
        tm.assert_series_equal(result, expected)

    @td.skip_array_manager_not_yet_implemented
    def test_reduction_timedelta_smallest_unit(self):
        df = DataFrame({'a': Series([pd.Timedelta('1 days')], dtype='timedelta64[s]'), 'b': Series([pd.Timedelta('1 days')], dtype='timedelta64[ms]')})
        result = df.max()
        expected = Series([pd.Timedelta('1 days'), pd.Timedelta('1 days')], dtype='timedelta64[ms]', index=['a', 'b'])
        tm.assert_series_equal(result, expected)