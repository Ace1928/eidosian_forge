from __future__ import annotations
from datetime import datetime
import pytest
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
from pandas.tseries.offsets import (
class TestQuarterBegin:

    def test_repr(self):
        expected = '<QuarterBegin: startingMonth=3>'
        assert repr(QuarterBegin()) == expected
        expected = '<QuarterBegin: startingMonth=3>'
        assert repr(QuarterBegin(startingMonth=3)) == expected
        expected = '<QuarterBegin: startingMonth=1>'
        assert repr(QuarterBegin(startingMonth=1)) == expected

    def test_is_anchored(self):
        msg = 'QuarterBegin.is_anchored is deprecated '
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert QuarterBegin(startingMonth=1).is_anchored()
            assert QuarterBegin().is_anchored()
            assert not QuarterBegin(2, startingMonth=1).is_anchored()

    def test_offset_corner_case(self):
        offset = QuarterBegin(n=-1, startingMonth=1)
        assert datetime(2010, 2, 1) + offset == datetime(2010, 1, 1)
    offset_cases = []
    offset_cases.append((QuarterBegin(startingMonth=1), {datetime(2007, 12, 1): datetime(2008, 1, 1), datetime(2008, 1, 1): datetime(2008, 4, 1), datetime(2008, 2, 15): datetime(2008, 4, 1), datetime(2008, 2, 29): datetime(2008, 4, 1), datetime(2008, 3, 15): datetime(2008, 4, 1), datetime(2008, 3, 31): datetime(2008, 4, 1), datetime(2008, 4, 15): datetime(2008, 7, 1), datetime(2008, 4, 1): datetime(2008, 7, 1)}))
    offset_cases.append((QuarterBegin(startingMonth=2), {datetime(2008, 1, 1): datetime(2008, 2, 1), datetime(2008, 1, 31): datetime(2008, 2, 1), datetime(2008, 1, 15): datetime(2008, 2, 1), datetime(2008, 2, 29): datetime(2008, 5, 1), datetime(2008, 3, 15): datetime(2008, 5, 1), datetime(2008, 3, 31): datetime(2008, 5, 1), datetime(2008, 4, 15): datetime(2008, 5, 1), datetime(2008, 4, 30): datetime(2008, 5, 1)}))
    offset_cases.append((QuarterBegin(startingMonth=1, n=0), {datetime(2008, 1, 1): datetime(2008, 1, 1), datetime(2008, 12, 1): datetime(2009, 1, 1), datetime(2008, 1, 1): datetime(2008, 1, 1), datetime(2008, 2, 15): datetime(2008, 4, 1), datetime(2008, 2, 29): datetime(2008, 4, 1), datetime(2008, 3, 15): datetime(2008, 4, 1), datetime(2008, 3, 31): datetime(2008, 4, 1), datetime(2008, 4, 15): datetime(2008, 7, 1), datetime(2008, 4, 30): datetime(2008, 7, 1)}))
    offset_cases.append((QuarterBegin(startingMonth=1, n=-1), {datetime(2008, 1, 1): datetime(2007, 10, 1), datetime(2008, 1, 31): datetime(2008, 1, 1), datetime(2008, 2, 15): datetime(2008, 1, 1), datetime(2008, 2, 29): datetime(2008, 1, 1), datetime(2008, 3, 15): datetime(2008, 1, 1), datetime(2008, 3, 31): datetime(2008, 1, 1), datetime(2008, 4, 15): datetime(2008, 4, 1), datetime(2008, 4, 30): datetime(2008, 4, 1), datetime(2008, 7, 1): datetime(2008, 4, 1)}))
    offset_cases.append((QuarterBegin(startingMonth=1, n=2), {datetime(2008, 1, 1): datetime(2008, 7, 1), datetime(2008, 2, 15): datetime(2008, 7, 1), datetime(2008, 2, 29): datetime(2008, 7, 1), datetime(2008, 3, 15): datetime(2008, 7, 1), datetime(2008, 3, 31): datetime(2008, 7, 1), datetime(2008, 4, 15): datetime(2008, 10, 1), datetime(2008, 4, 1): datetime(2008, 10, 1)}))

    @pytest.mark.parametrize('case', offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)