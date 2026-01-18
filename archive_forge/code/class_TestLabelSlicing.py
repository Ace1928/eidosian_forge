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
class TestLabelSlicing:

    def test_loc_getitem_slicing_datetimes_frame(self):
        df_unique = DataFrame(np.arange(4.0, dtype='float64'), index=[datetime(2001, 1, i, 10, 0) for i in [1, 2, 3, 4]])
        df_dups = DataFrame(np.arange(5.0, dtype='float64'), index=[datetime(2001, 1, i, 10, 0) for i in [1, 2, 2, 3, 4]])
        for df in [df_unique, df_dups]:
            result = df.loc[datetime(2001, 1, 1, 10):]
            tm.assert_frame_equal(result, df)
            result = df.loc[:datetime(2001, 1, 4, 10)]
            tm.assert_frame_equal(result, df)
            result = df.loc[datetime(2001, 1, 1, 10):datetime(2001, 1, 4, 10)]
            tm.assert_frame_equal(result, df)
            result = df.loc[datetime(2001, 1, 1, 11):]
            expected = df.iloc[1:]
            tm.assert_frame_equal(result, expected)
            result = df.loc['20010101 11':]
            tm.assert_frame_equal(result, expected)

    def test_loc_getitem_label_slice_across_dst(self):
        idx = date_range('2017-10-29 01:30:00', tz='Europe/Berlin', periods=5, freq='30 min')
        series2 = Series([0, 1, 2, 3, 4], index=idx)
        t_1 = Timestamp('2017-10-29 02:30:00+02:00', tz='Europe/Berlin')
        t_2 = Timestamp('2017-10-29 02:00:00+01:00', tz='Europe/Berlin')
        result = series2.loc[t_1:t_2]
        expected = Series([2, 3], index=idx[2:4])
        tm.assert_series_equal(result, expected)
        result = series2[t_1]
        expected = 2
        assert result == expected

    @pytest.mark.parametrize('index', [pd.period_range(start='2017-01-01', end='2018-01-01', freq='M'), timedelta_range(start='1 day', end='2 days', freq='1h')])
    def test_loc_getitem_label_slice_period_timedelta(self, index):
        ser = index.to_series()
        result = ser.loc[:index[-2]]
        expected = ser.iloc[:-1]
        tm.assert_series_equal(result, expected)

    def test_loc_getitem_slice_floats_inexact(self):
        index = [52195.504153, 52196.303147, 52198.369883]
        df = DataFrame(np.random.default_rng(2).random((3, 2)), index=index)
        s1 = df.loc[52195.1:52196.5]
        assert len(s1) == 2
        s1 = df.loc[52195.1:52196.6]
        assert len(s1) == 2
        s1 = df.loc[52195.1:52198.9]
        assert len(s1) == 3

    def test_loc_getitem_float_slice_floatindex(self, float_numpy_dtype):
        dtype = float_numpy_dtype
        ser = Series(np.random.default_rng(2).random(10), index=np.arange(10, 20, dtype=dtype))
        assert len(ser.loc[12.0:]) == 8
        assert len(ser.loc[12.5:]) == 7
        idx = np.arange(10, 20, dtype=dtype)
        idx[2] = 12.2
        ser.index = idx
        assert len(ser.loc[12.0:]) == 8
        assert len(ser.loc[12.5:]) == 7

    @pytest.mark.parametrize('start,stop, expected_slice', [[np.timedelta64(0, 'ns'), None, slice(0, 11)], [np.timedelta64(1, 'D'), np.timedelta64(6, 'D'), slice(1, 7)], [None, np.timedelta64(4, 'D'), slice(0, 5)]])
    def test_loc_getitem_slice_label_td64obj(self, start, stop, expected_slice):
        ser = Series(range(11), timedelta_range('0 days', '10 days'))
        result = ser.loc[slice(start, stop)]
        expected = ser.iloc[expected_slice]
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('start', ['2018', '2020'])
    def test_loc_getitem_slice_unordered_dt_index(self, frame_or_series, start):
        obj = frame_or_series([1, 2, 3], index=[Timestamp('2016'), Timestamp('2019'), Timestamp('2017')])
        with pytest.raises(KeyError, match='Value based partial slicing on non-monotonic'):
            obj.loc[start:'2022']

    @pytest.mark.parametrize('value', [1, 1.5])
    def test_loc_getitem_slice_labels_int_in_object_index(self, frame_or_series, value):
        obj = frame_or_series(range(4), index=[value, 'first', 2, 'third'])
        result = obj.loc[value:'third']
        expected = frame_or_series(range(4), index=[value, 'first', 2, 'third'])
        tm.assert_equal(result, expected)

    def test_loc_getitem_slice_columns_mixed_dtype(self):
        df = DataFrame({'test': 1, 1: 2, 2: 3}, index=[0])
        expected = DataFrame(data=[[2, 3]], index=[0], columns=Index([1, 2], dtype=object))
        tm.assert_frame_equal(df.loc[:, 1:], expected)