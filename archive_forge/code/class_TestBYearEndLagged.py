from __future__ import annotations
from datetime import datetime
import pytest
from pandas.tests.tseries.offsets.common import (
from pandas.tseries.offsets import (
class TestBYearEndLagged:

    def test_bad_month_fail(self):
        msg = 'Month must go from 1 to 12'
        with pytest.raises(ValueError, match=msg):
            BYearEnd(month=13)
        with pytest.raises(ValueError, match=msg):
            BYearEnd(month=0)
    offset_cases = []
    offset_cases.append((BYearEnd(month=6), {datetime(2008, 1, 1): datetime(2008, 6, 30), datetime(2007, 6, 30): datetime(2008, 6, 30)}))
    offset_cases.append((BYearEnd(n=-1, month=6), {datetime(2008, 1, 1): datetime(2007, 6, 29), datetime(2007, 6, 30): datetime(2007, 6, 29)}))

    @pytest.mark.parametrize('case', offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    def test_roll(self):
        offset = BYearEnd(month=6)
        date = datetime(2009, 11, 30)
        assert offset.rollforward(date) == datetime(2010, 6, 30)
        assert offset.rollback(date) == datetime(2009, 6, 30)
    on_offset_cases = [(BYearEnd(month=2), datetime(2007, 2, 28), True), (BYearEnd(month=6), datetime(2007, 6, 30), False)]

    @pytest.mark.parametrize('case', on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)