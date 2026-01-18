from datetime import (
import inspect
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
class TestReindexSetIndex:

    def test_dti_set_index_reindex_datetimeindex(self):
        df = DataFrame(np.random.default_rng(2).random(6))
        idx1 = date_range('2011/01/01', periods=6, freq='ME', tz='US/Eastern')
        idx2 = date_range('2013', periods=6, freq='YE', tz='Asia/Tokyo')
        df = df.set_index(idx1)
        tm.assert_index_equal(df.index, idx1)
        df = df.reindex(idx2)
        tm.assert_index_equal(df.index, idx2)

    def test_dti_set_index_reindex_freq_with_tz(self):
        index = date_range(datetime(2015, 10, 1), datetime(2015, 10, 1, 23), freq='h', tz='US/Eastern')
        df = DataFrame(np.random.default_rng(2).standard_normal((24, 1)), columns=['a'], index=index)
        new_index = date_range(datetime(2015, 10, 2), datetime(2015, 10, 2, 23), freq='h', tz='US/Eastern')
        result = df.set_index(new_index)
        assert result.index.freq == index.freq

    def test_set_reset_index_intervalindex(self):
        df = DataFrame({'A': range(10)})
        ser = pd.cut(df.A, 5)
        df['B'] = ser
        df = df.set_index('B')
        df = df.reset_index()

    def test_setitem_reset_index_dtypes(self):
        df = DataFrame(columns=['a', 'b', 'c']).astype({'a': 'datetime64[ns]', 'b': np.int64, 'c': np.float64})
        df1 = df.set_index(['a'])
        df1['d'] = []
        result = df1.reset_index()
        expected = DataFrame(columns=['a', 'b', 'c', 'd'], index=range(0)).astype({'a': 'datetime64[ns]', 'b': np.int64, 'c': np.float64, 'd': np.float64})
        tm.assert_frame_equal(result, expected)
        df2 = df.set_index(['a', 'b'])
        df2['d'] = []
        result = df2.reset_index()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('timezone, year, month, day, hour', [['America/Chicago', 2013, 11, 3, 1], ['America/Santiago', 2021, 4, 3, 23]])
    def test_reindex_timestamp_with_fold(self, timezone, year, month, day, hour):
        test_timezone = gettz(timezone)
        transition_1 = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=0, fold=0, tzinfo=test_timezone)
        transition_2 = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=0, fold=1, tzinfo=test_timezone)
        df = DataFrame({'index': [transition_1, transition_2], 'vals': ['a', 'b']}).set_index('index').reindex(['1', '2'])
        exp = DataFrame({'index': ['1', '2'], 'vals': [np.nan, np.nan]}).set_index('index')
        exp = exp.astype(df.vals.dtype)
        tm.assert_frame_equal(df, exp)