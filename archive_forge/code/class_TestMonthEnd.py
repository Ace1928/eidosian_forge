from __future__ import annotations
from datetime import datetime
import pytest
from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
from pandas import (
from pandas.tests.tseries.offsets.common import (
class TestMonthEnd:

    def test_day_of_month(self):
        dt = datetime(2007, 1, 1)
        offset = MonthEnd()
        result = dt + offset
        assert result == Timestamp(2007, 1, 31)
        result = result + offset
        assert result == Timestamp(2007, 2, 28)

    def test_normalize(self):
        dt = datetime(2007, 1, 1, 3)
        result = dt + MonthEnd(normalize=True)
        expected = dt.replace(hour=0) + MonthEnd()
        assert result == expected
    offset_cases = []
    offset_cases.append((MonthEnd(), {datetime(2008, 1, 1): datetime(2008, 1, 31), datetime(2008, 1, 31): datetime(2008, 2, 29), datetime(2006, 12, 29): datetime(2006, 12, 31), datetime(2006, 12, 31): datetime(2007, 1, 31), datetime(2007, 1, 1): datetime(2007, 1, 31), datetime(2006, 12, 1): datetime(2006, 12, 31)}))
    offset_cases.append((MonthEnd(0), {datetime(2008, 1, 1): datetime(2008, 1, 31), datetime(2008, 1, 31): datetime(2008, 1, 31), datetime(2006, 12, 29): datetime(2006, 12, 31), datetime(2006, 12, 31): datetime(2006, 12, 31), datetime(2007, 1, 1): datetime(2007, 1, 31)}))
    offset_cases.append((MonthEnd(2), {datetime(2008, 1, 1): datetime(2008, 2, 29), datetime(2008, 1, 31): datetime(2008, 3, 31), datetime(2006, 12, 29): datetime(2007, 1, 31), datetime(2006, 12, 31): datetime(2007, 2, 28), datetime(2007, 1, 1): datetime(2007, 2, 28), datetime(2006, 11, 1): datetime(2006, 12, 31)}))
    offset_cases.append((MonthEnd(-1), {datetime(2007, 1, 1): datetime(2006, 12, 31), datetime(2008, 6, 30): datetime(2008, 5, 31), datetime(2008, 12, 31): datetime(2008, 11, 30), datetime(2006, 12, 29): datetime(2006, 11, 30), datetime(2006, 12, 30): datetime(2006, 11, 30), datetime(2007, 1, 1): datetime(2006, 12, 31)}))

    @pytest.mark.parametrize('case', offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)
    on_offset_cases = [(MonthEnd(), datetime(2007, 12, 31), True), (MonthEnd(), datetime(2008, 1, 1), False)]

    @pytest.mark.parametrize('case', on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)