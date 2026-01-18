from __future__ import annotations
from datetime import datetime
import pytest
from pandas.tests.tseries.offsets.common import (
from pandas.tseries.offsets import (
class TestBYearBegin:

    def test_misspecified(self):
        msg = 'Month must go from 1 to 12'
        with pytest.raises(ValueError, match=msg):
            BYearBegin(month=13)
        with pytest.raises(ValueError, match=msg):
            BYearEnd(month=13)
    offset_cases = []
    offset_cases.append((BYearBegin(), {datetime(2008, 1, 1): datetime(2009, 1, 1), datetime(2008, 6, 30): datetime(2009, 1, 1), datetime(2008, 12, 31): datetime(2009, 1, 1), datetime(2011, 1, 1): datetime(2011, 1, 3), datetime(2011, 1, 3): datetime(2012, 1, 2), datetime(2005, 12, 30): datetime(2006, 1, 2), datetime(2005, 12, 31): datetime(2006, 1, 2)}))
    offset_cases.append((BYearBegin(0), {datetime(2008, 1, 1): datetime(2008, 1, 1), datetime(2008, 6, 30): datetime(2009, 1, 1), datetime(2008, 12, 31): datetime(2009, 1, 1), datetime(2005, 12, 30): datetime(2006, 1, 2), datetime(2005, 12, 31): datetime(2006, 1, 2)}))
    offset_cases.append((BYearBegin(-1), {datetime(2007, 1, 1): datetime(2006, 1, 2), datetime(2009, 1, 4): datetime(2009, 1, 1), datetime(2009, 1, 1): datetime(2008, 1, 1), datetime(2008, 6, 30): datetime(2008, 1, 1), datetime(2008, 12, 31): datetime(2008, 1, 1), datetime(2006, 12, 29): datetime(2006, 1, 2), datetime(2006, 12, 30): datetime(2006, 1, 2), datetime(2006, 1, 1): datetime(2005, 1, 3)}))
    offset_cases.append((BYearBegin(-2), {datetime(2007, 1, 1): datetime(2005, 1, 3), datetime(2007, 6, 30): datetime(2006, 1, 2), datetime(2008, 12, 31): datetime(2007, 1, 1)}))

    @pytest.mark.parametrize('case', offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)