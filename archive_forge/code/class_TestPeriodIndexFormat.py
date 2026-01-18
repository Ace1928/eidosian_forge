from contextlib import nullcontext
from datetime import (
import locale
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestPeriodIndexFormat:

    def test_period_format_and_strftime_default(self):
        per = PeriodIndex([datetime(2003, 1, 1, 12), None], freq='h')
        msg = 'PeriodIndex.format is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            formatted = per.format()
        assert formatted[0] == '2003-01-01 12:00'
        assert formatted[1] == 'NaT'
        assert formatted[0] == per.strftime(None)[0]
        assert per.strftime(None)[1] is np.nan
        per = pd.period_range('2003-01-01 12:01:01.123456789', periods=2, freq='ns')
        with tm.assert_produces_warning(FutureWarning, match=msg):
            formatted = per.format()
        assert (formatted == per.strftime(None)).all()
        assert formatted[0] == '2003-01-01 12:01:01.123456789'
        assert formatted[1] == '2003-01-01 12:01:01.123456790'

    def test_period_custom(self):
        msg = 'PeriodIndex.format is deprecated'
        per = pd.period_range('2003-01-01 12:01:01.123', periods=2, freq='ms')
        with tm.assert_produces_warning(FutureWarning, match=msg):
            formatted = per.format(date_format='%y %I:%M:%S (ms=%l us=%u ns=%n)')
        assert formatted[0] == '03 12:01:01 (ms=123 us=123000 ns=123000000)'
        assert formatted[1] == '03 12:01:01 (ms=124 us=124000 ns=124000000)'
        per = pd.period_range('2003-01-01 12:01:01.123456', periods=2, freq='us')
        with tm.assert_produces_warning(FutureWarning, match=msg):
            formatted = per.format(date_format='%y %I:%M:%S (ms=%l us=%u ns=%n)')
        assert formatted[0] == '03 12:01:01 (ms=123 us=123456 ns=123456000)'
        assert formatted[1] == '03 12:01:01 (ms=123 us=123457 ns=123457000)'
        per = pd.period_range('2003-01-01 12:01:01.123456789', periods=2, freq='ns')
        with tm.assert_produces_warning(FutureWarning, match=msg):
            formatted = per.format(date_format='%y %I:%M:%S (ms=%l us=%u ns=%n)')
        assert formatted[0] == '03 12:01:01 (ms=123 us=123456 ns=123456789)'
        assert formatted[1] == '03 12:01:01 (ms=123 us=123456 ns=123456790)'

    def test_period_tz(self):
        msg = 'PeriodIndex\\.format is deprecated'
        dt = pd.to_datetime(['2013-01-01 00:00:00+01:00'], utc=True)
        with tm.assert_produces_warning(UserWarning, match='will drop timezone'):
            per = dt.to_period(freq='h')
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert per.format()[0] == '2012-12-31 23:00'
        dt = dt.tz_convert('Europe/Paris')
        with tm.assert_produces_warning(UserWarning, match='will drop timezone'):
            per = dt.to_period(freq='h')
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert per.format()[0] == '2013-01-01 00:00'

    @pytest.mark.parametrize('locale_str', [pytest.param(None, id=str(locale.getlocale())), 'it_IT.utf8', 'it_IT', 'zh_CN.utf8', 'zh_CN'])
    def test_period_non_ascii_fmt(self, locale_str):
        if locale_str is not None and (not tm.can_set_locale(locale_str, locale.LC_ALL)):
            pytest.skip(f"Skipping as locale '{locale_str}' cannot be set on host.")
        with tm.set_locale(locale_str, locale.LC_ALL) if locale_str else nullcontext():
            per = pd.Period('2018-03-11 13:00', freq='h')
            assert per.strftime('%y é') == '18 é'
            per = pd.period_range('2003-01-01 01:00:00', periods=2, freq='12h')
            msg = 'PeriodIndex.format is deprecated'
            with tm.assert_produces_warning(FutureWarning, match=msg):
                formatted = per.format(date_format='%y é')
            assert formatted[0] == '03 é'
            assert formatted[1] == '03 é'

    @pytest.mark.parametrize('locale_str', [pytest.param(None, id=str(locale.getlocale())), 'it_IT.utf8', 'it_IT', 'zh_CN.utf8', 'zh_CN'])
    def test_period_custom_locale_directive(self, locale_str):
        if locale_str is not None and (not tm.can_set_locale(locale_str, locale.LC_ALL)):
            pytest.skip(f"Skipping as locale '{locale_str}' cannot be set on host.")
        with tm.set_locale(locale_str, locale.LC_ALL) if locale_str else nullcontext():
            am_local, pm_local = get_local_am_pm()
            per = pd.Period('2018-03-11 13:00', freq='h')
            assert per.strftime('%p') == pm_local
            per = pd.period_range('2003-01-01 01:00:00', periods=2, freq='12h')
            msg = 'PeriodIndex.format is deprecated'
            with tm.assert_produces_warning(FutureWarning, match=msg):
                formatted = per.format(date_format='%y %I:%M:%S%p')
            assert formatted[0] == f'03 01:00:00{am_local}'
            assert formatted[1] == f'03 01:00:00{pm_local}'