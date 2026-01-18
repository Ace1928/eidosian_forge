from __future__ import annotations
from datetime import datetime
import pytest
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
from pandas.tseries.offsets import (
class TestQuarterEnd:

    def test_repr(self):
        expected = '<QuarterEnd: startingMonth=3>'
        assert repr(QuarterEnd()) == expected
        expected = '<QuarterEnd: startingMonth=3>'
        assert repr(QuarterEnd(startingMonth=3)) == expected
        expected = '<QuarterEnd: startingMonth=1>'
        assert repr(QuarterEnd(startingMonth=1)) == expected

    def test_is_anchored(self):
        msg = 'QuarterEnd.is_anchored is deprecated '
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert QuarterEnd(startingMonth=1).is_anchored()
            assert QuarterEnd().is_anchored()
            assert not QuarterEnd(2, startingMonth=1).is_anchored()

    def test_offset_corner_case(self):
        offset = QuarterEnd(n=-1, startingMonth=1)
        assert datetime(2010, 2, 1) + offset == datetime(2010, 1, 31)
    offset_cases = []
    offset_cases.append((QuarterEnd(startingMonth=1), {datetime(2008, 1, 1): datetime(2008, 1, 31), datetime(2008, 1, 31): datetime(2008, 4, 30), datetime(2008, 2, 15): datetime(2008, 4, 30), datetime(2008, 2, 29): datetime(2008, 4, 30), datetime(2008, 3, 15): datetime(2008, 4, 30), datetime(2008, 3, 31): datetime(2008, 4, 30), datetime(2008, 4, 15): datetime(2008, 4, 30), datetime(2008, 4, 30): datetime(2008, 7, 31)}))
    offset_cases.append((QuarterEnd(startingMonth=2), {datetime(2008, 1, 1): datetime(2008, 2, 29), datetime(2008, 1, 31): datetime(2008, 2, 29), datetime(2008, 2, 15): datetime(2008, 2, 29), datetime(2008, 2, 29): datetime(2008, 5, 31), datetime(2008, 3, 15): datetime(2008, 5, 31), datetime(2008, 3, 31): datetime(2008, 5, 31), datetime(2008, 4, 15): datetime(2008, 5, 31), datetime(2008, 4, 30): datetime(2008, 5, 31)}))
    offset_cases.append((QuarterEnd(startingMonth=1, n=0), {datetime(2008, 1, 1): datetime(2008, 1, 31), datetime(2008, 1, 31): datetime(2008, 1, 31), datetime(2008, 2, 15): datetime(2008, 4, 30), datetime(2008, 2, 29): datetime(2008, 4, 30), datetime(2008, 3, 15): datetime(2008, 4, 30), datetime(2008, 3, 31): datetime(2008, 4, 30), datetime(2008, 4, 15): datetime(2008, 4, 30), datetime(2008, 4, 30): datetime(2008, 4, 30)}))
    offset_cases.append((QuarterEnd(startingMonth=1, n=-1), {datetime(2008, 1, 1): datetime(2007, 10, 31), datetime(2008, 1, 31): datetime(2007, 10, 31), datetime(2008, 2, 15): datetime(2008, 1, 31), datetime(2008, 2, 29): datetime(2008, 1, 31), datetime(2008, 3, 15): datetime(2008, 1, 31), datetime(2008, 3, 31): datetime(2008, 1, 31), datetime(2008, 4, 15): datetime(2008, 1, 31), datetime(2008, 4, 30): datetime(2008, 1, 31), datetime(2008, 7, 1): datetime(2008, 4, 30)}))
    offset_cases.append((QuarterEnd(startingMonth=1, n=2), {datetime(2008, 1, 31): datetime(2008, 7, 31), datetime(2008, 2, 15): datetime(2008, 7, 31), datetime(2008, 2, 29): datetime(2008, 7, 31), datetime(2008, 3, 15): datetime(2008, 7, 31), datetime(2008, 3, 31): datetime(2008, 7, 31), datetime(2008, 4, 15): datetime(2008, 7, 31), datetime(2008, 4, 30): datetime(2008, 10, 31)}))

    @pytest.mark.parametrize('case', offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)
    on_offset_cases = [(QuarterEnd(1, startingMonth=1), datetime(2008, 1, 31), True), (QuarterEnd(1, startingMonth=1), datetime(2007, 12, 31), False), (QuarterEnd(1, startingMonth=1), datetime(2008, 2, 29), False), (QuarterEnd(1, startingMonth=1), datetime(2007, 3, 30), False), (QuarterEnd(1, startingMonth=1), datetime(2007, 3, 31), False), (QuarterEnd(1, startingMonth=1), datetime(2008, 4, 30), True), (QuarterEnd(1, startingMonth=1), datetime(2008, 5, 30), False), (QuarterEnd(1, startingMonth=1), datetime(2008, 5, 31), False), (QuarterEnd(1, startingMonth=1), datetime(2007, 6, 29), False), (QuarterEnd(1, startingMonth=1), datetime(2007, 6, 30), False), (QuarterEnd(1, startingMonth=2), datetime(2008, 1, 31), False), (QuarterEnd(1, startingMonth=2), datetime(2007, 12, 31), False), (QuarterEnd(1, startingMonth=2), datetime(2008, 2, 29), True), (QuarterEnd(1, startingMonth=2), datetime(2007, 3, 30), False), (QuarterEnd(1, startingMonth=2), datetime(2007, 3, 31), False), (QuarterEnd(1, startingMonth=2), datetime(2008, 4, 30), False), (QuarterEnd(1, startingMonth=2), datetime(2008, 5, 30), False), (QuarterEnd(1, startingMonth=2), datetime(2008, 5, 31), True), (QuarterEnd(1, startingMonth=2), datetime(2007, 6, 29), False), (QuarterEnd(1, startingMonth=2), datetime(2007, 6, 30), False), (QuarterEnd(1, startingMonth=3), datetime(2008, 1, 31), False), (QuarterEnd(1, startingMonth=3), datetime(2007, 12, 31), True), (QuarterEnd(1, startingMonth=3), datetime(2008, 2, 29), False), (QuarterEnd(1, startingMonth=3), datetime(2007, 3, 30), False), (QuarterEnd(1, startingMonth=3), datetime(2007, 3, 31), True), (QuarterEnd(1, startingMonth=3), datetime(2008, 4, 30), False), (QuarterEnd(1, startingMonth=3), datetime(2008, 5, 30), False), (QuarterEnd(1, startingMonth=3), datetime(2008, 5, 31), False), (QuarterEnd(1, startingMonth=3), datetime(2007, 6, 29), False), (QuarterEnd(1, startingMonth=3), datetime(2007, 6, 30), True)]

    @pytest.mark.parametrize('case', on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)