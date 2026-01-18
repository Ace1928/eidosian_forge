from __future__ import annotations
from datetime import datetime
import pytest
from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
from pandas import (
from pandas.tests.tseries.offsets.common import (
class TestMonthBegin:
    offset_cases = []
    offset_cases.append((MonthBegin(), {datetime(2008, 1, 31): datetime(2008, 2, 1), datetime(2008, 2, 1): datetime(2008, 3, 1), datetime(2006, 12, 31): datetime(2007, 1, 1), datetime(2006, 12, 1): datetime(2007, 1, 1), datetime(2007, 1, 31): datetime(2007, 2, 1)}))
    offset_cases.append((MonthBegin(0), {datetime(2008, 1, 31): datetime(2008, 2, 1), datetime(2008, 1, 1): datetime(2008, 1, 1), datetime(2006, 12, 3): datetime(2007, 1, 1), datetime(2007, 1, 31): datetime(2007, 2, 1)}))
    offset_cases.append((MonthBegin(2), {datetime(2008, 2, 29): datetime(2008, 4, 1), datetime(2008, 1, 31): datetime(2008, 3, 1), datetime(2006, 12, 31): datetime(2007, 2, 1), datetime(2007, 12, 28): datetime(2008, 2, 1), datetime(2007, 1, 1): datetime(2007, 3, 1), datetime(2006, 11, 1): datetime(2007, 1, 1)}))
    offset_cases.append((MonthBegin(-1), {datetime(2007, 1, 1): datetime(2006, 12, 1), datetime(2008, 5, 31): datetime(2008, 5, 1), datetime(2008, 12, 31): datetime(2008, 12, 1), datetime(2006, 12, 29): datetime(2006, 12, 1), datetime(2006, 1, 2): datetime(2006, 1, 1)}))

    @pytest.mark.parametrize('case', offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)