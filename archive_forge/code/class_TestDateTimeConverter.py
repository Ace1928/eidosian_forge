from datetime import (
import subprocess
import sys
import numpy as np
import pytest
import pandas._config.config as cf
from pandas._libs.tslibs import to_offset
from pandas import (
import pandas._testing as tm
from pandas.plotting import (
from pandas.tseries.offsets import (
class TestDateTimeConverter:

    @pytest.fixture
    def dtc(self):
        return converter.DatetimeConverter()

    def test_convert_accepts_unicode(self, dtc):
        r1 = dtc.convert('2000-01-01 12:22', None, None)
        r2 = dtc.convert('2000-01-01 12:22', None, None)
        assert r1 == r2, 'DatetimeConverter.convert should accept unicode'

    def test_conversion(self, dtc):
        rs = dtc.convert(['2012-1-1'], None, None)[0]
        xp = dates.date2num(datetime(2012, 1, 1))
        assert rs == xp
        rs = dtc.convert('2012-1-1', None, None)
        assert rs == xp
        rs = dtc.convert(date(2012, 1, 1), None, None)
        assert rs == xp
        rs = dtc.convert('2012-1-1', None, None)
        assert rs == xp
        rs = dtc.convert(Timestamp('2012-1-1'), None, None)
        assert rs == xp
        rs = dtc.convert('2012-01-01', None, None)
        assert rs == xp
        rs = dtc.convert('2012-01-01 00:00:00+0000', None, None)
        assert rs == xp
        rs = dtc.convert(np.array(['2012-01-01 00:00:00+0000', '2012-01-02 00:00:00+0000']), None, None)
        assert rs[0] == xp
        ts = Timestamp('2012-01-01').tz_localize('UTC').tz_convert('US/Eastern')
        rs = dtc.convert(ts, None, None)
        assert rs == xp
        rs = dtc.convert(ts.to_pydatetime(), None, None)
        assert rs == xp
        rs = dtc.convert(Index([ts - Day(1), ts]), None, None)
        assert rs[1] == xp
        rs = dtc.convert(Index([ts - Day(1), ts]).to_pydatetime(), None, None)
        assert rs[1] == xp

    def test_conversion_float(self, dtc):
        rtol = 0.5 * 10 ** (-9)
        rs = dtc.convert(Timestamp('2012-1-1 01:02:03', tz='UTC'), None, None)
        xp = converter.mdates.date2num(Timestamp('2012-1-1 01:02:03', tz='UTC'))
        tm.assert_almost_equal(rs, xp, rtol=rtol)
        rs = dtc.convert(Timestamp('2012-1-1 09:02:03', tz='Asia/Hong_Kong'), None, None)
        tm.assert_almost_equal(rs, xp, rtol=rtol)
        rs = dtc.convert(datetime(2012, 1, 1, 1, 2, 3), None, None)
        tm.assert_almost_equal(rs, xp, rtol=rtol)

    @pytest.mark.parametrize('values', [[date(1677, 1, 1), date(1677, 1, 2)], [datetime(1677, 1, 1, 12), datetime(1677, 1, 2, 12)]])
    def test_conversion_outofbounds_datetime(self, dtc, values):
        rs = dtc.convert(values, None, None)
        xp = converter.mdates.date2num(values)
        tm.assert_numpy_array_equal(rs, xp)
        rs = dtc.convert(values[0], None, None)
        xp = converter.mdates.date2num(values[0])
        assert rs == xp

    @pytest.mark.parametrize('time,format_expected', [(0, '00:00'), (86399.999999, '23:59:59.999999'), (90000, '01:00'), (3723, '01:02:03'), (39723.2, '11:02:03.200')])
    def test_time_formatter(self, time, format_expected):
        result = converter.TimeFormatter(None)(time)
        assert result == format_expected

    @pytest.mark.parametrize('freq', ('B', 'ms', 's'))
    def test_dateindex_conversion(self, freq, dtc):
        rtol = 10 ** (-9)
        dateindex = date_range('2020-01-01', periods=10, freq=freq)
        rs = dtc.convert(dateindex, None, None)
        xp = converter.mdates.date2num(dateindex._mpl_repr())
        tm.assert_almost_equal(rs, xp, rtol=rtol)

    @pytest.mark.parametrize('offset', [Second(), Milli(), Micro(50)])
    def test_resolution(self, offset, dtc):
        ts1 = Timestamp('2012-1-1')
        ts2 = ts1 + offset
        val1 = dtc.convert(ts1, None, None)
        val2 = dtc.convert(ts2, None, None)
        if not val1 < val2:
            raise AssertionError(f'{val1} is not less than {val2}.')

    def test_convert_nested(self, dtc):
        inner = [Timestamp('2017-01-01'), Timestamp('2017-01-02')]
        data = [inner, inner]
        result = dtc.convert(data, None, None)
        expected = [dtc.convert(x, None, None) for x in data]
        assert (np.array(result) == expected).all()