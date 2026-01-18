from datetime import (
import re
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas._libs.tslibs.parsing import DateParseError
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
class TestPeriodMethods:

    def test_round_trip(self):
        p = Period('2000Q1')
        new_p = tm.round_trip_pickle(p)
        assert new_p == p

    def test_hash(self):
        assert hash(Period('2011-01', freq='M')) == hash(Period('2011-01', freq='M'))
        assert hash(Period('2011-01-01', freq='D')) != hash(Period('2011-01', freq='M'))
        assert hash(Period('2011-01', freq='3M')) != hash(Period('2011-01', freq='2M'))
        assert hash(Period('2011-01', freq='M')) != hash(Period('2011-02', freq='M'))

    def test_to_timestamp_mult(self):
        p = Period('2011-01', freq='M')
        assert p.to_timestamp(how='S') == Timestamp('2011-01-01')
        expected = Timestamp('2011-02-01') - Timedelta(1, 'ns')
        assert p.to_timestamp(how='E') == expected
        p = Period('2011-01', freq='3M')
        assert p.to_timestamp(how='S') == Timestamp('2011-01-01')
        expected = Timestamp('2011-04-01') - Timedelta(1, 'ns')
        assert p.to_timestamp(how='E') == expected

    @pytest.mark.filterwarnings('ignore:Period with BDay freq is deprecated:FutureWarning')
    def test_to_timestamp(self):
        p = Period('1982', freq='Y')
        start_ts = p.to_timestamp(how='S')
        aliases = ['s', 'StarT', 'BEGIn']
        for a in aliases:
            assert start_ts == p.to_timestamp('D', how=a)
            assert start_ts == p.to_timestamp('3D', how=a)
        end_ts = p.to_timestamp(how='E')
        aliases = ['e', 'end', 'FINIsH']
        for a in aliases:
            assert end_ts == p.to_timestamp('D', how=a)
            assert end_ts == p.to_timestamp('3D', how=a)
        from_lst = ['Y', 'Q', 'M', 'W', 'B', 'D', 'h', 'Min', 's']

        def _ex(p):
            if p.freq == 'B':
                return p.start_time + Timedelta(days=1, nanoseconds=-1)
            return Timestamp((p + p.freq).start_time._value - 1)
        for fcode in from_lst:
            p = Period('1982', freq=fcode)
            result = p.to_timestamp().to_period(fcode)
            assert result == p
            assert p.start_time == p.to_timestamp(how='S')
            assert p.end_time == _ex(p)
        p = Period('1985', freq='Y')
        result = p.to_timestamp('h', how='end')
        expected = Timestamp(1986, 1, 1) - Timedelta(1, 'ns')
        assert result == expected
        result = p.to_timestamp('3h', how='end')
        assert result == expected
        result = p.to_timestamp('min', how='end')
        expected = Timestamp(1986, 1, 1) - Timedelta(1, 'ns')
        assert result == expected
        result = p.to_timestamp('2min', how='end')
        assert result == expected
        result = p.to_timestamp(how='end')
        expected = Timestamp(1986, 1, 1) - Timedelta(1, 'ns')
        assert result == expected
        expected = datetime(1985, 1, 1)
        result = p.to_timestamp('h', how='start')
        assert result == expected
        result = p.to_timestamp('min', how='start')
        assert result == expected
        result = p.to_timestamp('s', how='start')
        assert result == expected
        result = p.to_timestamp('3h', how='start')
        assert result == expected
        result = p.to_timestamp('5s', how='start')
        assert result == expected

    def test_to_timestamp_business_end(self):
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            per = Period('1990-01-05', 'B')
            result = per.to_timestamp('B', how='E')
        expected = Timestamp('1990-01-06') - Timedelta(nanoseconds=1)
        assert result == expected

    @pytest.mark.parametrize('ts, expected', [('1970-01-01 00:00:00', 0), ('1970-01-01 00:00:00.000001', 1), ('1970-01-01 00:00:00.00001', 10), ('1970-01-01 00:00:00.499', 499000), ('1999-12-31 23:59:59.999', 999000), ('1999-12-31 23:59:59.999999', 999999), ('2050-12-31 23:59:59.5', 500000), ('2050-12-31 23:59:59.500001', 500001), ('2050-12-31 23:59:59.123456', 123456)])
    @pytest.mark.parametrize('freq', [None, 'us', 'ns'])
    def test_to_timestamp_microsecond(self, ts, expected, freq):
        result = Period(ts).to_timestamp(freq=freq).microsecond
        assert result == expected

    @pytest.mark.parametrize('str_ts,freq,str_res,str_freq', (('Jan-2000', None, '2000-01', 'M'), ('2000-12-15', None, '2000-12-15', 'D'), ('2000-12-15 13:45:26.123456789', 'ns', '2000-12-15 13:45:26.123456789', 'ns'), ('2000-12-15 13:45:26.123456789', 'us', '2000-12-15 13:45:26.123456', 'us'), ('2000-12-15 13:45:26.123456', None, '2000-12-15 13:45:26.123456', 'us'), ('2000-12-15 13:45:26.123456789', 'ms', '2000-12-15 13:45:26.123', 'ms'), ('2000-12-15 13:45:26.123', None, '2000-12-15 13:45:26.123', 'ms'), ('2000-12-15 13:45:26', 's', '2000-12-15 13:45:26', 's'), ('2000-12-15 13:45:26', 'min', '2000-12-15 13:45', 'min'), ('2000-12-15 13:45:26', 'h', '2000-12-15 13:00', 'h'), ('2000-12-15', 'Y', '2000', 'Y-DEC'), ('2000-12-15', 'Q', '2000Q4', 'Q-DEC'), ('2000-12-15', 'M', '2000-12', 'M'), ('2000-12-15', 'W', '2000-12-11/2000-12-17', 'W-SUN'), ('2000-12-15', 'D', '2000-12-15', 'D'), ('2000-12-15', 'B', '2000-12-15', 'B')))
    @pytest.mark.filterwarnings('ignore:Period with BDay freq is deprecated:FutureWarning')
    def test_repr(self, str_ts, freq, str_res, str_freq):
        p = Period(str_ts, freq=freq)
        assert str(p) == str_res
        assert repr(p) == f"Period('{str_res}', '{str_freq}')"

    def test_repr_nat(self):
        p = Period('nat', freq='M')
        assert repr(NaT) in repr(p)

    def test_strftime(self):
        p = Period('2000-1-1 12:34:12', freq='s')
        res = p.strftime('%Y-%m-%d %H:%M:%S')
        assert res == '2000-01-01 12:34:12'
        assert isinstance(res, str)