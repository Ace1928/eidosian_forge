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
class TestDatetimeParsingWrappers:

    @pytest.mark.parametrize('date_str, expected', [('2011-01-01', datetime(2011, 1, 1)), ('2Q2005', datetime(2005, 4, 1)), ('2Q05', datetime(2005, 4, 1)), ('2005Q1', datetime(2005, 1, 1)), ('05Q1', datetime(2005, 1, 1)), ('2011Q3', datetime(2011, 7, 1)), ('11Q3', datetime(2011, 7, 1)), ('3Q2011', datetime(2011, 7, 1)), ('3Q11', datetime(2011, 7, 1)), ('2000Q4', datetime(2000, 10, 1)), ('00Q4', datetime(2000, 10, 1)), ('4Q2000', datetime(2000, 10, 1)), ('4Q00', datetime(2000, 10, 1)), ('2000q4', datetime(2000, 10, 1)), ('2000-Q4', datetime(2000, 10, 1)), ('00-Q4', datetime(2000, 10, 1)), ('4Q-2000', datetime(2000, 10, 1)), ('4Q-00', datetime(2000, 10, 1)), ('00q4', datetime(2000, 10, 1)), ('2005', datetime(2005, 1, 1)), ('2005-11', datetime(2005, 11, 1)), ('2005 11', datetime(2005, 11, 1)), ('11-2005', datetime(2005, 11, 1)), ('11 2005', datetime(2005, 11, 1)), ('200511', datetime(2020, 5, 11)), ('20051109', datetime(2005, 11, 9)), ('20051109 10:15', datetime(2005, 11, 9, 10, 15)), ('20051109 08H', datetime(2005, 11, 9, 8, 0)), ('2005-11-09 10:15', datetime(2005, 11, 9, 10, 15)), ('2005-11-09 08H', datetime(2005, 11, 9, 8, 0)), ('2005/11/09 10:15', datetime(2005, 11, 9, 10, 15)), ('2005/11/09 10:15:32', datetime(2005, 11, 9, 10, 15, 32)), ('2005/11/09 10:15:32 AM', datetime(2005, 11, 9, 10, 15, 32)), ('2005/11/09 10:15:32 PM', datetime(2005, 11, 9, 22, 15, 32)), ('2005/11/09 08H', datetime(2005, 11, 9, 8, 0)), ('Thu Sep 25 10:36:28 2003', datetime(2003, 9, 25, 10, 36, 28)), ('Thu Sep 25 2003', datetime(2003, 9, 25)), ('Sep 25 2003', datetime(2003, 9, 25)), ('January 1 2014', datetime(2014, 1, 1)), ('2014-06', datetime(2014, 6, 1)), ('06-2014', datetime(2014, 6, 1)), ('2014-6', datetime(2014, 6, 1)), ('6-2014', datetime(2014, 6, 1)), ('20010101 12', datetime(2001, 1, 1, 12)), ('20010101 1234', datetime(2001, 1, 1, 12, 34)), ('20010101 123456', datetime(2001, 1, 1, 12, 34, 56))])
    def test_parsers(self, date_str, expected, cache):
        yearfirst = True
        result1, _ = parsing.parse_datetime_string_with_reso(date_str, yearfirst=yearfirst)
        result2 = to_datetime(date_str, yearfirst=yearfirst)
        result3 = to_datetime([date_str], yearfirst=yearfirst)
        result4 = to_datetime(np.array([date_str], dtype=object), yearfirst=yearfirst, cache=cache)
        result6 = DatetimeIndex([date_str], yearfirst=yearfirst)
        result8 = DatetimeIndex(Index([date_str]), yearfirst=yearfirst)
        result9 = DatetimeIndex(Series([date_str]), yearfirst=yearfirst)
        for res in [result1, result2]:
            assert res == expected
        for res in [result3, result4, result6, result8, result9]:
            exp = DatetimeIndex([Timestamp(expected)])
            tm.assert_index_equal(res, exp)
        if not yearfirst:
            result5 = Timestamp(date_str)
            assert result5 == expected
            result7 = date_range(date_str, freq='S', periods=1, yearfirst=yearfirst)
            assert result7 == expected

    def test_na_values_with_cache(self, cache, unique_nulls_fixture, unique_nulls_fixture2):
        expected = Index([NaT, NaT], dtype='datetime64[ns]')
        result = to_datetime([unique_nulls_fixture, unique_nulls_fixture2], cache=cache)
        tm.assert_index_equal(result, expected)

    def test_parsers_nat(self):
        result1, _ = parsing.parse_datetime_string_with_reso('NaT')
        result2 = to_datetime('NaT')
        result3 = Timestamp('NaT')
        result4 = DatetimeIndex(['NaT'])[0]
        assert result1 is NaT
        assert result2 is NaT
        assert result3 is NaT
        assert result4 is NaT

    @pytest.mark.parametrize('date_str, dayfirst, yearfirst, expected', [('10-11-12', False, False, datetime(2012, 10, 11)), ('10-11-12', True, False, datetime(2012, 11, 10)), ('10-11-12', False, True, datetime(2010, 11, 12)), ('10-11-12', True, True, datetime(2010, 12, 11)), ('20/12/21', False, False, datetime(2021, 12, 20)), ('20/12/21', True, False, datetime(2021, 12, 20)), ('20/12/21', False, True, datetime(2020, 12, 21)), ('20/12/21', True, True, datetime(2020, 12, 21))])
    def test_parsers_dayfirst_yearfirst(self, cache, date_str, dayfirst, yearfirst, expected):
        dateutil_result = parse(date_str, dayfirst=dayfirst, yearfirst=yearfirst)
        assert dateutil_result == expected
        result1, _ = parsing.parse_datetime_string_with_reso(date_str, dayfirst=dayfirst, yearfirst=yearfirst)
        if not dayfirst and (not yearfirst):
            result2 = Timestamp(date_str)
            assert result2 == expected
        result3 = to_datetime(date_str, dayfirst=dayfirst, yearfirst=yearfirst, cache=cache)
        result4 = DatetimeIndex([date_str], dayfirst=dayfirst, yearfirst=yearfirst)[0]
        assert result1 == expected
        assert result3 == expected
        assert result4 == expected

    @pytest.mark.parametrize('date_str, exp_def', [['10:15', datetime(1, 1, 1, 10, 15)], ['9:05', datetime(1, 1, 1, 9, 5)]])
    def test_parsers_timestring(self, date_str, exp_def):
        exp_now = parse(date_str)
        result1, _ = parsing.parse_datetime_string_with_reso(date_str)
        result2 = to_datetime(date_str)
        result3 = to_datetime([date_str])
        result4 = Timestamp(date_str)
        result5 = DatetimeIndex([date_str])[0]
        assert result1 == exp_def
        assert result2 == exp_now
        assert result3 == exp_now
        assert result4 == exp_now
        assert result5 == exp_now

    @pytest.mark.parametrize('dt_string, tz, dt_string_repr', [('2013-01-01 05:45+0545', timezone(timedelta(minutes=345)), "Timestamp('2013-01-01 05:45:00+0545', tz='UTC+05:45')"), ('2013-01-01 05:30+0530', timezone(timedelta(minutes=330)), "Timestamp('2013-01-01 05:30:00+0530', tz='UTC+05:30')")])
    def test_parsers_timezone_minute_offsets_roundtrip(self, cache, dt_string, tz, dt_string_repr):
        base = to_datetime('2013-01-01 00:00:00', cache=cache)
        base = base.tz_localize('UTC').tz_convert(tz)
        dt_time = to_datetime(dt_string, cache=cache)
        assert base == dt_time
        assert dt_string_repr == repr(dt_time)