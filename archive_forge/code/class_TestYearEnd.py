from __future__ import annotations
from datetime import datetime
import numpy as np
import pytest
from pandas import Timestamp
from pandas.tests.tseries.offsets.common import (
from pandas.tseries.offsets import (
class TestYearEnd:

    def test_misspecified(self):
        with pytest.raises(ValueError, match='Month must go from 1 to 12'):
            YearEnd(month=13)
    offset_cases = []
    offset_cases.append((YearEnd(), {datetime(2008, 1, 1): datetime(2008, 12, 31), datetime(2008, 6, 30): datetime(2008, 12, 31), datetime(2008, 12, 31): datetime(2009, 12, 31), datetime(2005, 12, 30): datetime(2005, 12, 31), datetime(2005, 12, 31): datetime(2006, 12, 31)}))
    offset_cases.append((YearEnd(0), {datetime(2008, 1, 1): datetime(2008, 12, 31), datetime(2008, 6, 30): datetime(2008, 12, 31), datetime(2008, 12, 31): datetime(2008, 12, 31), datetime(2005, 12, 30): datetime(2005, 12, 31)}))
    offset_cases.append((YearEnd(-1), {datetime(2007, 1, 1): datetime(2006, 12, 31), datetime(2008, 6, 30): datetime(2007, 12, 31), datetime(2008, 12, 31): datetime(2007, 12, 31), datetime(2006, 12, 29): datetime(2005, 12, 31), datetime(2006, 12, 30): datetime(2005, 12, 31), datetime(2007, 1, 1): datetime(2006, 12, 31)}))
    offset_cases.append((YearEnd(-2), {datetime(2007, 1, 1): datetime(2005, 12, 31), datetime(2008, 6, 30): datetime(2006, 12, 31), datetime(2008, 12, 31): datetime(2006, 12, 31)}))

    @pytest.mark.parametrize('case', offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)
    on_offset_cases = [(YearEnd(), datetime(2007, 12, 31), True), (YearEnd(), datetime(2008, 1, 1), False), (YearEnd(), datetime(2006, 12, 31), True), (YearEnd(), datetime(2006, 12, 29), False)]

    @pytest.mark.parametrize('case', on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)