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
class TestOrigin:

    def test_origin_and_unit(self):
        ts = to_datetime(1, unit='s', origin=1)
        expected = Timestamp('1970-01-01 00:00:02')
        assert ts == expected
        ts = to_datetime(1, unit='s', origin=1000000000)
        expected = Timestamp('2001-09-09 01:46:41')
        assert ts == expected

    def test_julian(self, julian_dates):
        result = Series(to_datetime(julian_dates, unit='D', origin='julian'))
        expected = Series(to_datetime(julian_dates - Timestamp(0).to_julian_date(), unit='D'))
        tm.assert_series_equal(result, expected)

    def test_unix(self):
        result = Series(to_datetime([0, 1, 2], unit='D', origin='unix'))
        expected = Series([Timestamp('1970-01-01'), Timestamp('1970-01-02'), Timestamp('1970-01-03')], dtype='M8[ns]')
        tm.assert_series_equal(result, expected)

    def test_julian_round_trip(self):
        result = to_datetime(2456658, origin='julian', unit='D')
        assert result.to_julian_date() == 2456658
        msg = "1 is Out of Bounds for origin='julian'"
        with pytest.raises(ValueError, match=msg):
            to_datetime(1, origin='julian', unit='D')

    def test_invalid_unit(self, units, julian_dates):
        if units != 'D':
            msg = "unit must be 'D' for origin='julian'"
            with pytest.raises(ValueError, match=msg):
                to_datetime(julian_dates, unit=units, origin='julian')

    @pytest.mark.parametrize('unit', ['ns', 'D'])
    def test_invalid_origin(self, unit):
        msg = 'it must be numeric with a unit specified'
        with pytest.raises(ValueError, match=msg):
            to_datetime('2005-01-01', origin='1960-01-01', unit=unit)

    def test_epoch(self, units, epochs, epoch_1960, units_from_epochs):
        expected = Series([pd.Timedelta(x, unit=units) + epoch_1960 for x in units_from_epochs])
        result = Series(to_datetime(units_from_epochs, unit=units, origin=epochs))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('origin, exc', [('random_string', ValueError), ('epoch', ValueError), ('13-24-1990', ValueError), (datetime(1, 1, 1), OutOfBoundsDatetime)])
    def test_invalid_origins(self, origin, exc, units, units_from_epochs):
        msg = '|'.join([f'origin {origin} is Out of Bounds', f'origin {origin} cannot be converted to a Timestamp', "Cannot cast .* to unit='ns' without overflow"])
        with pytest.raises(exc, match=msg):
            to_datetime(units_from_epochs, unit=units, origin=origin)

    def test_invalid_origins_tzinfo(self):
        with pytest.raises(ValueError, match='must be tz-naive'):
            to_datetime(1, unit='D', origin=datetime(2000, 1, 1, tzinfo=pytz.utc))

    def test_incorrect_value_exception(self):
        msg = 'Unknown datetime string format, unable to parse: yesterday, at position 1'
        with pytest.raises(ValueError, match=msg):
            to_datetime(['today', 'yesterday'])

    @pytest.mark.parametrize('format, warning', [(None, UserWarning), ('%Y-%m-%d %H:%M:%S', None), ('%Y-%d-%m %H:%M:%S', None)])
    def test_to_datetime_out_of_bounds_with_format_arg(self, format, warning):
        msg = '^Out of bounds nanosecond timestamp: 2417-10-10 00:00:00, at position 0'
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime('2417-10-10 00:00:00', format=format)

    @pytest.mark.parametrize('arg, origin, expected_str', [[200 * 365, 'unix', '2169-11-13 00:00:00'], [200 * 365, '1870-01-01', '2069-11-13 00:00:00'], [300 * 365, '1870-01-01', '2169-10-20 00:00:00']])
    def test_processing_order(self, arg, origin, expected_str):
        result = to_datetime(arg, unit='D', origin=origin)
        expected = Timestamp(expected_str)
        assert result == expected
        result = to_datetime(200 * 365, unit='D', origin='1870-01-01')
        expected = Timestamp('2069-11-13 00:00:00')
        assert result == expected
        result = to_datetime(300 * 365, unit='D', origin='1870-01-01')
        expected = Timestamp('2169-10-20 00:00:00')
        assert result == expected

    @pytest.mark.parametrize('offset,utc,exp', [['Z', True, '2019-01-01T00:00:00.000Z'], ['Z', None, '2019-01-01T00:00:00.000Z'], ['-01:00', True, '2019-01-01T01:00:00.000Z'], ['-01:00', None, '2019-01-01T00:00:00.000-01:00']])
    def test_arg_tz_ns_unit(self, offset, utc, exp):
        arg = '2019-01-01T00:00:00.000' + offset
        result = to_datetime([arg], unit='ns', utc=utc)
        expected = to_datetime([exp]).as_unit('ns')
        tm.assert_index_equal(result, expected)