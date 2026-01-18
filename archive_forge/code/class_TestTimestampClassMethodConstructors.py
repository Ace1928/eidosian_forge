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
class TestTimestampClassMethodConstructors:

    def test_constructor_strptime(self):
        fmt = '%Y%m%d-%H%M%S-%f%z'
        ts = '20190129-235348-000001+0000'
        msg = 'Timestamp.strptime\\(\\) is not implemented'
        with pytest.raises(NotImplementedError, match=msg):
            Timestamp.strptime(ts, fmt)

    def test_constructor_fromisocalendar(self):
        expected_timestamp = Timestamp('2000-01-03 00:00:00')
        expected_stdlib = datetime.fromisocalendar(2000, 1, 1)
        result = Timestamp.fromisocalendar(2000, 1, 1)
        assert result == expected_timestamp
        assert result == expected_stdlib
        assert isinstance(result, Timestamp)

    def test_constructor_fromordinal(self):
        base = datetime(2000, 1, 1)
        ts = Timestamp.fromordinal(base.toordinal())
        assert base == ts
        assert base.toordinal() == ts.toordinal()
        ts = Timestamp.fromordinal(base.toordinal(), tz='US/Eastern')
        assert Timestamp('2000-01-01', tz='US/Eastern') == ts
        assert base.toordinal() == ts.toordinal()
        dt = datetime(2011, 4, 16, 0, 0)
        ts = Timestamp.fromordinal(dt.toordinal())
        assert ts.to_pydatetime() == dt
        stamp = Timestamp('2011-4-16', tz='US/Eastern')
        dt_tz = stamp.to_pydatetime()
        ts = Timestamp.fromordinal(dt_tz.toordinal(), tz='US/Eastern')
        assert ts.to_pydatetime() == dt_tz

    def test_now(self):
        ts_from_string = Timestamp('now')
        ts_from_method = Timestamp.now()
        ts_datetime = datetime.now()
        ts_from_string_tz = Timestamp('now', tz='US/Eastern')
        ts_from_method_tz = Timestamp.now(tz='US/Eastern')
        delta = Timedelta(seconds=1)
        assert abs(ts_from_method - ts_from_string) < delta
        assert abs(ts_datetime - ts_from_method) < delta
        assert abs(ts_from_method_tz - ts_from_string_tz) < delta
        assert abs(ts_from_string_tz.tz_localize(None) - ts_from_method_tz.tz_localize(None)) < delta

    def test_today(self):
        ts_from_string = Timestamp('today')
        ts_from_method = Timestamp.today()
        ts_datetime = datetime.today()
        ts_from_string_tz = Timestamp('today', tz='US/Eastern')
        ts_from_method_tz = Timestamp.today(tz='US/Eastern')
        delta = Timedelta(seconds=1)
        assert abs(ts_from_method - ts_from_string) < delta
        assert abs(ts_datetime - ts_from_method) < delta
        assert abs(ts_from_method_tz - ts_from_string_tz) < delta
        assert abs(ts_from_string_tz.tz_localize(None) - ts_from_method_tz.tz_localize(None)) < delta