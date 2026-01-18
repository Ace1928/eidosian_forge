from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
from pandas.tseries import offsets
class TestCustomBusinessMonthEnd:

    @pytest.fixture
    def _offset(self):
        return CBMonthEnd

    @pytest.fixture
    def offset(self):
        return CBMonthEnd()

    @pytest.fixture
    def offset2(self):
        return CBMonthEnd(2)

    def test_different_normalize_equals(self, _offset):
        offset = _offset()
        offset2 = _offset(normalize=True)
        assert offset != offset2

    def test_repr(self, offset, offset2):
        assert repr(offset) == '<CustomBusinessMonthEnd>'
        assert repr(offset2) == '<2 * CustomBusinessMonthEnds>'

    def test_add_datetime(self, dt, offset2):
        assert offset2 + dt == datetime(2008, 2, 29)

    def testRollback1(self):
        assert CDay(10).rollback(datetime(2007, 12, 31)) == datetime(2007, 12, 31)

    def testRollback2(self, dt):
        assert CBMonthEnd(10).rollback(dt) == datetime(2007, 12, 31)

    def testRollforward1(self, dt):
        assert CBMonthEnd(10).rollforward(dt) == datetime(2008, 1, 31)

    def test_roll_date_object(self):
        offset = CBMonthEnd()
        dt = date(2012, 9, 15)
        result = offset.rollback(dt)
        assert result == datetime(2012, 8, 31)
        result = offset.rollforward(dt)
        assert result == datetime(2012, 9, 28)
        offset = offsets.Day()
        result = offset.rollback(dt)
        assert result == datetime(2012, 9, 15)
        result = offset.rollforward(dt)
        assert result == datetime(2012, 9, 15)
    on_offset_cases = [(CBMonthEnd(), datetime(2008, 1, 31), True), (CBMonthEnd(), datetime(2008, 1, 1), False)]

    @pytest.mark.parametrize('case', on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)
    apply_cases = [(CBMonthEnd(), {datetime(2008, 1, 1): datetime(2008, 1, 31), datetime(2008, 2, 7): datetime(2008, 2, 29)}), (2 * CBMonthEnd(), {datetime(2008, 1, 1): datetime(2008, 2, 29), datetime(2008, 2, 7): datetime(2008, 3, 31)}), (-CBMonthEnd(), {datetime(2008, 1, 1): datetime(2007, 12, 31), datetime(2008, 2, 8): datetime(2008, 1, 31)}), (-2 * CBMonthEnd(), {datetime(2008, 1, 1): datetime(2007, 11, 30), datetime(2008, 2, 9): datetime(2007, 12, 31)}), (CBMonthEnd(0), {datetime(2008, 1, 1): datetime(2008, 1, 31), datetime(2008, 2, 7): datetime(2008, 2, 29)})]

    @pytest.mark.parametrize('case', apply_cases)
    def test_apply(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    def test_apply_large_n(self):
        dt = datetime(2012, 10, 23)
        result = dt + CBMonthEnd(10)
        assert result == datetime(2013, 7, 31)
        result = dt + CDay(100) - CDay(100)
        assert result == dt
        off = CBMonthEnd() * 6
        rs = datetime(2012, 1, 1) - off
        xp = datetime(2011, 7, 29)
        assert rs == xp
        st = datetime(2011, 12, 18)
        rs = st + off
        xp = datetime(2012, 5, 31)
        assert rs == xp

    def test_holidays(self):
        holidays = ['2012-01-31', datetime(2012, 2, 28), np.datetime64('2012-02-29')]
        bm_offset = CBMonthEnd(holidays=holidays)
        dt = datetime(2012, 1, 1)
        assert dt + bm_offset == datetime(2012, 1, 30)
        assert dt + 2 * bm_offset == datetime(2012, 2, 27)

    @pytest.mark.parametrize('case', [(CBMonthEnd(n=1, offset=timedelta(days=5)), {datetime(2021, 3, 1): datetime(2021, 3, 31) + timedelta(days=5), datetime(2021, 4, 17): datetime(2021, 4, 30) + timedelta(days=5)}), (CBMonthEnd(n=2, offset=timedelta(days=40)), {datetime(2021, 3, 10): datetime(2021, 4, 30) + timedelta(days=40), datetime(2021, 4, 30): datetime(2021, 6, 30) + timedelta(days=40)}), (CBMonthEnd(n=1, offset=timedelta(days=-5)), {datetime(2021, 3, 1): datetime(2021, 3, 31) - timedelta(days=5), datetime(2021, 4, 11): datetime(2021, 4, 30) - timedelta(days=5)}), (-2 * CBMonthEnd(n=1, offset=timedelta(days=10)), {datetime(2021, 3, 1): datetime(2021, 1, 29) + timedelta(days=10), datetime(2021, 4, 3): datetime(2021, 2, 26) + timedelta(days=10)}), (CBMonthEnd(n=0, offset=timedelta(days=1)), {datetime(2021, 3, 2): datetime(2021, 3, 31) + timedelta(days=1), datetime(2021, 4, 1): datetime(2021, 4, 30) + timedelta(days=1)}), (CBMonthEnd(n=1, holidays=['2021-03-31'], offset=timedelta(days=1)), {datetime(2021, 3, 2): datetime(2021, 3, 30) + timedelta(days=1)})])
    def test_apply_with_extra_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)