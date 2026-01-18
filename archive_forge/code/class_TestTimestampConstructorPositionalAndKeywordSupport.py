import calendar
from datetime import (
import zoneinfo
import dateutil.tz
from dateutil.tz import (
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.compat import PY310
from pandas.errors import OutOfBoundsDatetime
from pandas import (
class TestTimestampConstructorPositionalAndKeywordSupport:

    def test_constructor_positional(self):
        msg = "'NoneType' object cannot be interpreted as an integer" if PY310 else 'an integer is required'
        with pytest.raises(TypeError, match=msg):
            Timestamp(2000, 1)
        msg = 'month must be in 1..12'
        with pytest.raises(ValueError, match=msg):
            Timestamp(2000, 0, 1)
        with pytest.raises(ValueError, match=msg):
            Timestamp(2000, 13, 1)
        msg = 'day is out of range for month'
        with pytest.raises(ValueError, match=msg):
            Timestamp(2000, 1, 0)
        with pytest.raises(ValueError, match=msg):
            Timestamp(2000, 1, 32)
        assert repr(Timestamp(2015, 11, 12)) == repr(Timestamp('20151112'))
        assert repr(Timestamp(2015, 11, 12, 1, 2, 3, 999999)) == repr(Timestamp('2015-11-12 01:02:03.999999'))

    def test_constructor_keyword(self):
        msg = "function missing required argument 'day'|Required argument 'day'"
        with pytest.raises(TypeError, match=msg):
            Timestamp(year=2000, month=1)
        msg = 'month must be in 1..12'
        with pytest.raises(ValueError, match=msg):
            Timestamp(year=2000, month=0, day=1)
        with pytest.raises(ValueError, match=msg):
            Timestamp(year=2000, month=13, day=1)
        msg = 'day is out of range for month'
        with pytest.raises(ValueError, match=msg):
            Timestamp(year=2000, month=1, day=0)
        with pytest.raises(ValueError, match=msg):
            Timestamp(year=2000, month=1, day=32)
        assert repr(Timestamp(year=2015, month=11, day=12)) == repr(Timestamp('20151112'))
        assert repr(Timestamp(year=2015, month=11, day=12, hour=1, minute=2, second=3, microsecond=999999)) == repr(Timestamp('2015-11-12 01:02:03.999999'))

    @pytest.mark.parametrize('arg', ['year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond', 'nanosecond'])
    def test_invalid_date_kwarg_with_string_input(self, arg):
        kwarg = {arg: 1}
        msg = 'Cannot pass a date attribute keyword argument'
        with pytest.raises(ValueError, match=msg):
            Timestamp('2010-10-10 12:59:59.999999999', **kwarg)

    @pytest.mark.parametrize('kwargs', [{}, {'year': 2020}, {'year': 2020, 'month': 1}])
    def test_constructor_missing_keyword(self, kwargs):
        msg1 = "function missing required argument '(year|month|day)' \\(pos [123]\\)"
        msg2 = "Required argument '(year|month|day)' \\(pos [123]\\) not found"
        msg = '|'.join([msg1, msg2])
        with pytest.raises(TypeError, match=msg):
            Timestamp(**kwargs)

    def test_constructor_positional_with_tzinfo(self):
        ts = Timestamp(2020, 12, 31, tzinfo=timezone.utc)
        expected = Timestamp('2020-12-31', tzinfo=timezone.utc)
        assert ts == expected

    @pytest.mark.parametrize('kwd', ['nanosecond', 'microsecond', 'second', 'minute'])
    def test_constructor_positional_keyword_mixed_with_tzinfo(self, kwd, request):
        if kwd != 'nanosecond':
            mark = pytest.mark.xfail(reason='GH#45307')
            request.applymarker(mark)
        kwargs = {kwd: 4}
        ts = Timestamp(2020, 12, 31, tzinfo=timezone.utc, **kwargs)
        td_kwargs = {kwd + 's': 4}
        td = Timedelta(**td_kwargs)
        expected = Timestamp('2020-12-31', tz=timezone.utc) + td
        assert ts == expected