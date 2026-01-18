import calendar
from collections import deque
from datetime import (
from decimal import Decimal
import locale
from dateutil.parser import parse
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
import pytz
from pandas._libs import tslib
from pandas._libs.tslibs import (
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_datetime64_ns_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
from pandas.core.tools import datetimes as tools
from pandas.core.tools.datetimes import start_caching_at
class TestToDatetimeDataFrame:

    @pytest.fixture
    def df(self):
        return DataFrame({'year': [2015, 2016], 'month': [2, 3], 'day': [4, 5], 'hour': [6, 7], 'minute': [58, 59], 'second': [10, 11], 'ms': [1, 1], 'us': [2, 2], 'ns': [3, 3]})

    def test_dataframe(self, df, cache):
        result = to_datetime({'year': df['year'], 'month': df['month'], 'day': df['day']}, cache=cache)
        expected = Series([Timestamp('20150204 00:00:00'), Timestamp('20160305 00:0:00')])
        tm.assert_series_equal(result, expected)
        result = to_datetime(df[['year', 'month', 'day']].to_dict(), cache=cache)
        tm.assert_series_equal(result, expected)

    def test_dataframe_dict_with_constructable(self, df, cache):
        df2 = df[['year', 'month', 'day']].to_dict()
        df2['month'] = 2
        result = to_datetime(df2, cache=cache)
        expected2 = Series([Timestamp('20150204 00:00:00'), Timestamp('20160205 00:0:00')])
        tm.assert_series_equal(result, expected2)

    @pytest.mark.parametrize('unit', [{'year': 'years', 'month': 'months', 'day': 'days', 'hour': 'hours', 'minute': 'minutes', 'second': 'seconds'}, {'year': 'year', 'month': 'month', 'day': 'day', 'hour': 'hour', 'minute': 'minute', 'second': 'second'}])
    def test_dataframe_field_aliases_column_subset(self, df, cache, unit):
        result = to_datetime(df[list(unit.keys())].rename(columns=unit), cache=cache)
        expected = Series([Timestamp('20150204 06:58:10'), Timestamp('20160305 07:59:11')], dtype='M8[ns]')
        tm.assert_series_equal(result, expected)

    def test_dataframe_field_aliases(self, df, cache):
        d = {'year': 'year', 'month': 'month', 'day': 'day', 'hour': 'hour', 'minute': 'minute', 'second': 'second', 'ms': 'ms', 'us': 'us', 'ns': 'ns'}
        result = to_datetime(df.rename(columns=d), cache=cache)
        expected = Series([Timestamp('20150204 06:58:10.001002003'), Timestamp('20160305 07:59:11.001002003')])
        tm.assert_series_equal(result, expected)

    def test_dataframe_str_dtype(self, df, cache):
        result = to_datetime(df.astype(str), cache=cache)
        expected = Series([Timestamp('20150204 06:58:10.001002003'), Timestamp('20160305 07:59:11.001002003')])
        tm.assert_series_equal(result, expected)

    def test_dataframe_coerce(self, cache):
        df2 = DataFrame({'year': [2015, 2016], 'month': [2, 20], 'day': [4, 5]})
        msg = '^cannot assemble the datetimes: time data ".+" doesn\\\'t match format "%Y%m%d", at position 1\\.'
        with pytest.raises(ValueError, match=msg):
            to_datetime(df2, cache=cache)
        result = to_datetime(df2, errors='coerce', cache=cache)
        expected = Series([Timestamp('20150204 00:00:00'), NaT])
        tm.assert_series_equal(result, expected)

    def test_dataframe_extra_keys_raisesm(self, df, cache):
        msg = 'extra keys have been passed to the datetime assemblage: \\[foo\\]'
        with pytest.raises(ValueError, match=msg):
            df2 = df.copy()
            df2['foo'] = 1
            to_datetime(df2, cache=cache)

    @pytest.mark.parametrize('cols', [['year'], ['year', 'month'], ['year', 'month', 'second'], ['month', 'day'], ['year', 'day', 'second']])
    def test_dataframe_missing_keys_raises(self, df, cache, cols):
        msg = 'to assemble mappings requires at least that \\[year, month, day\\] be specified: \\[.+\\] is missing'
        with pytest.raises(ValueError, match=msg):
            to_datetime(df[cols], cache=cache)

    def test_dataframe_duplicate_columns_raises(self, cache):
        msg = 'cannot assemble with duplicate keys'
        df2 = DataFrame({'year': [2015, 2016], 'month': [2, 20], 'day': [4, 5]})
        df2.columns = ['year', 'year', 'day']
        with pytest.raises(ValueError, match=msg):
            to_datetime(df2, cache=cache)
        df2 = DataFrame({'year': [2015, 2016], 'month': [2, 20], 'day': [4, 5], 'hour': [4, 5]})
        df2.columns = ['year', 'month', 'day', 'day']
        with pytest.raises(ValueError, match=msg):
            to_datetime(df2, cache=cache)

    def test_dataframe_int16(self, cache):
        df = DataFrame({'year': [2015, 2016], 'month': [2, 3], 'day': [4, 5]})
        result = to_datetime(df.astype('int16'), cache=cache)
        expected = Series([Timestamp('20150204 00:00:00'), Timestamp('20160305 00:00:00')])
        tm.assert_series_equal(result, expected)

    def test_dataframe_mixed(self, cache):
        df = DataFrame({'year': [2015, 2016], 'month': [2, 3], 'day': [4, 5]})
        df['month'] = df['month'].astype('int8')
        df['day'] = df['day'].astype('int8')
        result = to_datetime(df, cache=cache)
        expected = Series([Timestamp('20150204 00:00:00'), Timestamp('20160305 00:00:00')])
        tm.assert_series_equal(result, expected)

    def test_dataframe_float(self, cache):
        df = DataFrame({'year': [2000, 2001], 'month': [1.5, 1], 'day': [1, 1]})
        msg = '^cannot assemble the datetimes: unconverted data remains when parsing with format ".*": "1", at position 0.'
        with pytest.raises(ValueError, match=msg):
            to_datetime(df, cache=cache)

    def test_dataframe_utc_true(self):
        df = DataFrame({'year': [2015, 2016], 'month': [2, 3], 'day': [4, 5]})
        result = to_datetime(df, utc=True)
        expected = Series(np.array(['2015-02-04', '2016-03-05'], dtype='datetime64[ns]')).dt.tz_localize('UTC')
        tm.assert_series_equal(result, expected)