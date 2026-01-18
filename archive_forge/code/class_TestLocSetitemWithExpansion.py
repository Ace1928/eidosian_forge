from collections import namedtuple
from datetime import (
import re
from dateutil.tz import gettz
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import index as libindex
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.core.indexing import _one_ellipsis_message
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
class TestLocSetitemWithExpansion:

    def test_loc_setitem_with_expansion_large_dataframe(self, monkeypatch):
        size_cutoff = 50
        with monkeypatch.context():
            monkeypatch.setattr(libindex, '_SIZE_CUTOFF', size_cutoff)
            result = DataFrame({'x': range(size_cutoff)}, dtype='int64')
            result.loc[size_cutoff] = size_cutoff
        expected = DataFrame({'x': range(size_cutoff + 1)}, dtype='int64')
        tm.assert_frame_equal(result, expected)

    def test_loc_setitem_empty_series(self):
        ser = Series(dtype=object)
        ser.loc[1] = 1
        tm.assert_series_equal(ser, Series([1], index=[1]))
        ser.loc[3] = 3
        tm.assert_series_equal(ser, Series([1, 3], index=[1, 3]))

    def test_loc_setitem_empty_series_float(self):
        ser = Series(dtype=object)
        ser.loc[1] = 1.0
        tm.assert_series_equal(ser, Series([1.0], index=[1]))
        ser.loc[3] = 3.0
        tm.assert_series_equal(ser, Series([1.0, 3.0], index=[1, 3]))

    def test_loc_setitem_empty_series_str_idx(self):
        ser = Series(dtype=object)
        ser.loc['foo'] = 1
        tm.assert_series_equal(ser, Series([1], index=Index(['foo'], dtype=object)))
        ser.loc['bar'] = 3
        tm.assert_series_equal(ser, Series([1, 3], index=Index(['foo', 'bar'], dtype=object)))
        ser.loc[3] = 4
        tm.assert_series_equal(ser, Series([1, 3, 4], index=Index(['foo', 'bar', 3], dtype=object)))

    def test_loc_setitem_incremental_with_dst(self):
        base = datetime(2015, 11, 1, tzinfo=gettz('US/Pacific'))
        idxs = [base + timedelta(seconds=i * 900) for i in range(16)]
        result = Series([0], index=[idxs[0]])
        for ts in idxs:
            result.loc[ts] = 1
        expected = Series(1, index=idxs)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('conv', [lambda x: x, lambda x: x.to_datetime64(), lambda x: x.to_pydatetime(), lambda x: np.datetime64(x)], ids=['self', 'to_datetime64', 'to_pydatetime', 'np.datetime64'])
    def test_loc_setitem_datetime_keys_cast(self, conv):
        dt1 = Timestamp('20130101 09:00:00')
        dt2 = Timestamp('20130101 10:00:00')
        df = DataFrame()
        df.loc[conv(dt1), 'one'] = 100
        df.loc[conv(dt2), 'one'] = 200
        expected = DataFrame({'one': [100.0, 200.0]}, index=[dt1, dt2], columns=Index(['one'], dtype=object))
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_categorical_column_retains_dtype(self, ordered):
        result = DataFrame({'A': [1]})
        result.loc[:, 'B'] = Categorical(['b'], ordered=ordered)
        expected = DataFrame({'A': [1], 'B': Categorical(['b'], ordered=ordered)})
        tm.assert_frame_equal(result, expected)

    def test_loc_setitem_with_expansion_and_existing_dst(self):
        start = Timestamp('2017-10-29 00:00:00+0200', tz='Europe/Madrid')
        end = Timestamp('2017-10-29 03:00:00+0100', tz='Europe/Madrid')
        ts = Timestamp('2016-10-10 03:00:00', tz='Europe/Madrid')
        idx = date_range(start, end, inclusive='left', freq='h')
        assert ts not in idx
        result = DataFrame(index=idx, columns=['value'])
        result.loc[ts, 'value'] = 12
        expected = DataFrame([np.nan] * len(idx) + [12], index=idx.append(DatetimeIndex([ts])), columns=['value'], dtype=object)
        tm.assert_frame_equal(result, expected)

    def test_setitem_with_expansion(self):
        df = DataFrame(data=to_datetime(['2015-03-30 20:12:32', '2015-03-12 00:11:11']), columns=['time'])
        df['new_col'] = ['new', 'old']
        df.time = df.set_index('time').index.tz_localize('UTC')
        v = df[df.new_col == 'new'].set_index('time').index.tz_convert('US/Pacific')
        df2 = df.copy()
        df2.loc[df2.new_col == 'new', 'time'] = v
        expected = Series([v[0].tz_convert('UTC'), df.loc[1, 'time']], name='time')
        tm.assert_series_equal(df2.time, expected)
        v = df.loc[df.new_col == 'new', 'time'] + Timedelta('1s')
        df.loc[df.new_col == 'new', 'time'] = v
        tm.assert_series_equal(df.loc[df.new_col == 'new', 'time'], v)

    def test_loc_setitem_with_expansion_inf_upcast_empty(self):
        df = DataFrame()
        df.loc[0, 0] = 1
        df.loc[1, 1] = 2
        df.loc[0, np.inf] = 3
        result = df.columns
        expected = Index([0, 1, np.inf], dtype=np.float64)
        tm.assert_index_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:indexing past lexsort depth')
    def test_loc_setitem_with_expansion_nonunique_index(self, index):
        if not len(index):
            pytest.skip('Not relevant for empty Index')
        index = index.repeat(2)
        N = len(index)
        arr = np.arange(N).astype(np.int64)
        orig = DataFrame(arr, index=index, columns=[0])
        key = 'kapow'
        assert key not in index
        exp_index = index.insert(len(index), key)
        if isinstance(index, MultiIndex):
            assert exp_index[-1][0] == key
        else:
            assert exp_index[-1] == key
        exp_data = np.arange(N + 1).astype(np.float64)
        expected = DataFrame(exp_data, index=exp_index, columns=[0])
        df = orig.copy()
        df.loc[key, 0] = N
        tm.assert_frame_equal(df, expected)
        ser = orig.copy()[0]
        ser.loc[key] = N
        expected = expected[0].astype(np.int64)
        tm.assert_series_equal(ser, expected)
        df = orig.copy()
        df.loc[key, 1] = N
        expected = DataFrame({0: list(arr) + [np.nan], 1: [np.nan] * N + [float(N)]}, index=exp_index)
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize('dtype', ['Int32', 'Int64', 'UInt32', 'UInt64', 'Float32', 'Float64'])
    def test_loc_setitem_with_expansion_preserves_nullable_int(self, dtype):
        ser = Series([0, 1, 2, 3], dtype=dtype)
        df = DataFrame({'data': ser})
        result = DataFrame(index=df.index)
        result.loc[df.index, 'data'] = ser
        tm.assert_frame_equal(result, df, check_column_type=False)
        result = DataFrame(index=df.index)
        result.loc[df.index, 'data'] = ser._values
        tm.assert_frame_equal(result, df, check_column_type=False)

    def test_loc_setitem_ea_not_full_column(self):
        df = DataFrame({'A': range(5)})
        val = date_range('2016-01-01', periods=3, tz='US/Pacific')
        df.loc[[0, 1, 2], 'B'] = val
        bex = val.append(DatetimeIndex([pd.NaT, pd.NaT], dtype=val.dtype))
        expected = DataFrame({'A': range(5), 'B': bex})
        assert expected.dtypes['B'] == val.dtype
        tm.assert_frame_equal(df, expected)