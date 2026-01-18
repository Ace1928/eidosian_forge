from datetime import timedelta
import pytest
import pytz
from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
from pandas.errors import PerformanceWarning
from pandas import DatetimeIndex
import pandas._testing as tm
from pandas.util.version import Version
def _test_offset(self, offset_name, offset_n, tstart, expected_utc_offset):
    offset = DateOffset(**{offset_name: offset_n})
    if offset_name in ['hour', 'minute', 'second', 'microsecond'] and offset_n == 1 and (tstart == Timestamp('2013-11-03 01:59:59.999999-0500', tz='US/Eastern')):
        err_msg = {'hour': '2013-11-03 01:59:59.999999', 'minute': '2013-11-03 01:01:59.999999', 'second': '2013-11-03 01:59:01.999999', 'microsecond': '2013-11-03 01:59:59.000001'}[offset_name]
        with pytest.raises(pytz.AmbiguousTimeError, match=err_msg):
            tstart + offset
        dti = DatetimeIndex([tstart])
        warn_msg = 'Non-vectorized DateOffset'
        with pytest.raises(pytz.AmbiguousTimeError, match=err_msg):
            with tm.assert_produces_warning(PerformanceWarning, match=warn_msg):
                dti + offset
        return
    t = tstart + offset
    if expected_utc_offset is not None:
        assert get_utc_offset_hours(t) == expected_utc_offset
    if offset_name == 'weeks':
        assert t.date() == timedelta(days=7 * offset.kwds['weeks']) + tstart.date()
        assert t.dayofweek == tstart.dayofweek and t.hour == tstart.hour and (t.minute == tstart.minute) and (t.second == tstart.second)
    elif offset_name == 'days':
        assert timedelta(offset.kwds['days']) + tstart.date() == t.date()
        assert t.hour == tstart.hour and t.minute == tstart.minute and (t.second == tstart.second)
    elif offset_name in self.valid_date_offsets_singular:
        datepart_offset = getattr(t, offset_name if offset_name != 'weekday' else 'dayofweek')
        assert datepart_offset == offset.kwds[offset_name]
    else:
        assert t == (tstart.tz_convert('UTC') + offset).tz_convert('US/Pacific')