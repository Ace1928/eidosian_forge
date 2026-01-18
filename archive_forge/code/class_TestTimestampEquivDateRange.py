from datetime import (
import re
import numpy as np
import pytest
import pytz
from pytz import timezone
from pandas._libs.tslibs import timezones
from pandas._libs.tslibs.offsets import (
from pandas.errors import OutOfBoundsDatetime
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.datetimes import _generate_range as generate_range
from pandas.tests.indexes.datetimes.test_timezones import (
from pandas.tseries.holiday import USFederalHolidayCalendar
class TestTimestampEquivDateRange:

    def test_date_range_timestamp_equiv(self):
        rng = date_range('20090415', '20090519', tz='US/Eastern')
        stamp = rng[0]
        ts = Timestamp('20090415', tz='US/Eastern')
        assert ts == stamp

    def test_date_range_timestamp_equiv_dateutil(self):
        rng = date_range('20090415', '20090519', tz='dateutil/US/Eastern')
        stamp = rng[0]
        ts = Timestamp('20090415', tz='dateutil/US/Eastern')
        assert ts == stamp

    def test_date_range_timestamp_equiv_explicit_pytz(self):
        rng = date_range('20090415', '20090519', tz=pytz.timezone('US/Eastern'))
        stamp = rng[0]
        ts = Timestamp('20090415', tz=pytz.timezone('US/Eastern'))
        assert ts == stamp

    @td.skip_if_windows
    def test_date_range_timestamp_equiv_explicit_dateutil(self):
        from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
        rng = date_range('20090415', '20090519', tz=gettz('US/Eastern'))
        stamp = rng[0]
        ts = Timestamp('20090415', tz=gettz('US/Eastern'))
        assert ts == stamp

    def test_date_range_timestamp_equiv_from_datetime_instance(self):
        datetime_instance = datetime(2014, 3, 4)
        timestamp_instance = date_range(datetime_instance, periods=1, freq='D')[0]
        ts = Timestamp(datetime_instance)
        assert ts == timestamp_instance

    def test_date_range_timestamp_equiv_preserve_frequency(self):
        timestamp_instance = date_range('2014-03-05', periods=1, freq='D')[0]
        ts = Timestamp('2014-03-05')
        assert timestamp_instance == ts