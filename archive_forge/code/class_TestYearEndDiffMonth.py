from __future__ import annotations
from datetime import datetime
import numpy as np
import pytest
from pandas import Timestamp
from pandas.tests.tseries.offsets.common import (
from pandas.tseries.offsets import (
class TestYearEndDiffMonth:
    offset_cases = []
    offset_cases.append((YearEnd(month=3), {datetime(2008, 1, 1): datetime(2008, 3, 31), datetime(2008, 2, 15): datetime(2008, 3, 31), datetime(2008, 3, 31): datetime(2009, 3, 31), datetime(2008, 3, 30): datetime(2008, 3, 31), datetime(2005, 3, 31): datetime(2006, 3, 31), datetime(2006, 7, 30): datetime(2007, 3, 31)}))
    offset_cases.append((YearEnd(0, month=3), {datetime(2008, 1, 1): datetime(2008, 3, 31), datetime(2008, 2, 28): datetime(2008, 3, 31), datetime(2008, 3, 31): datetime(2008, 3, 31), datetime(2005, 3, 30): datetime(2005, 3, 31)}))
    offset_cases.append((YearEnd(-1, month=3), {datetime(2007, 1, 1): datetime(2006, 3, 31), datetime(2008, 2, 28): datetime(2007, 3, 31), datetime(2008, 3, 31): datetime(2007, 3, 31), datetime(2006, 3, 29): datetime(2005, 3, 31), datetime(2006, 3, 30): datetime(2005, 3, 31), datetime(2007, 3, 1): datetime(2006, 3, 31)}))
    offset_cases.append((YearEnd(-2, month=3), {datetime(2007, 1, 1): datetime(2005, 3, 31), datetime(2008, 6, 30): datetime(2007, 3, 31), datetime(2008, 3, 31): datetime(2006, 3, 31)}))

    @pytest.mark.parametrize('case', offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)
    on_offset_cases = [(YearEnd(month=3), datetime(2007, 3, 31), True), (YearEnd(month=3), datetime(2008, 1, 1), False), (YearEnd(month=3), datetime(2006, 3, 31), True), (YearEnd(month=3), datetime(2006, 3, 29), False)]

    @pytest.mark.parametrize('case', on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)