from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
from pandas.tseries import offsets
class TestCustomBusinessMonthBegin:

    @pytest.fixture
    def _offset(self):
        return CBMonthBegin

    @pytest.fixture
    def offset(self):
        return CBMonthBegin()

    @pytest.fixture
    def offset2(self):
        return CBMonthBegin(2)

    def test_different_normalize_equals(self, _offset):
        offset = _offset()
        offset2 = _offset(normalize=True)
        assert offset != offset2

    def test_repr(self, offset, offset2):
        assert repr(offset) == '<CustomBusinessMonthBegin>'
        assert repr(offset2) == '<2 * CustomBusinessMonthBegins>'

    def test_add_datetime(self, dt, offset2):
        assert offset2 + dt == datetime(2008, 3, 3)

    def testRollback1(self):
        assert CDay(10).rollback(datetime(2007, 12, 31)) == datetime(2007, 12, 31)

    def testRollback2(self, dt):
        assert CBMonthBegin(10).rollback(dt) == datetime(2008, 1, 1)

    def testRollforward1(self, dt):
        assert CBMonthBegin(10).rollforward(dt) == datetime(2008, 1, 1)

    def test_roll_date_object(self):
        offset = CBMonthBegin()
        dt = date(2012, 9, 15)
        result = offset.rollback(dt)
        assert result == datetime(2012, 9, 3)
        result = offset.rollforward(dt)
        assert result == datetime(2012, 10, 1)
        offset = offsets.Day()
        result = offset.rollback(dt)
        assert result == datetime(2012, 9, 15)
        result = offset.rollforward(dt)
        assert result == datetime(2012, 9, 15)
    on_offset_cases = [(CBMonthBegin(), datetime(2008, 1, 1), True), (CBMonthBegin(), datetime(2008, 1, 31), False)]

    @pytest.mark.parametrize('case', on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)
    apply_cases = [(CBMonthBegin(), {datetime(2008, 1, 1): datetime(2008, 2, 1), datetime(2008, 2, 7): datetime(2008, 3, 3)}), (2 * CBMonthBegin(), {datetime(2008, 1, 1): datetime(2008, 3, 3), datetime(2008, 2, 7): datetime(2008, 4, 1)}), (-CBMonthBegin(), {datetime(2008, 1, 1): datetime(2007, 12, 3), datetime(2008, 2, 8): datetime(2008, 2, 1)}), (-2 * CBMonthBegin(), {datetime(2008, 1, 1): datetime(2007, 11, 1), datetime(2008, 2, 9): datetime(2008, 1, 1)}), (CBMonthBegin(0), {datetime(2008, 1, 1): datetime(2008, 1, 1), datetime(2008, 1, 7): datetime(2008, 2, 1)})]

    @pytest.mark.parametrize('case', apply_cases)
    def test_apply(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    def test_apply_large_n(self):
        dt = datetime(2012, 10, 23)
        result = dt + CBMonthBegin(10)
        assert result == datetime(2013, 8, 1)
        result = dt + CDay(100) - CDay(100)
        assert result == dt
        off = CBMonthBegin() * 6
        rs = datetime(2012, 1, 1) - off
        xp = datetime(2011, 7, 1)
        assert rs == xp
        st = datetime(2011, 12, 18)
        rs = st + off
        xp = datetime(2012, 6, 1)
        assert rs == xp

    def test_holidays(self):
        holidays = ['2012-02-01', datetime(2012, 2, 2), np.datetime64('2012-03-01')]
        bm_offset = CBMonthBegin(holidays=holidays)
        dt = datetime(2012, 1, 1)
        assert dt + bm_offset == datetime(2012, 1, 2)
        assert dt + 2 * bm_offset == datetime(2012, 2, 3)

    @pytest.mark.parametrize('case', [(CBMonthBegin(n=1, offset=timedelta(days=5)), {datetime(2021, 3, 1): datetime(2021, 4, 1) + timedelta(days=5), datetime(2021, 4, 17): datetime(2021, 5, 3) + timedelta(days=5)}), (CBMonthBegin(n=2, offset=timedelta(days=40)), {datetime(2021, 3, 10): datetime(2021, 5, 3) + timedelta(days=40), datetime(2021, 4, 30): datetime(2021, 6, 1) + timedelta(days=40)}), (CBMonthBegin(n=1, offset=timedelta(days=-5)), {datetime(2021, 3, 1): datetime(2021, 4, 1) - timedelta(days=5), datetime(2021, 4, 11): datetime(2021, 5, 3) - timedelta(days=5)}), (-2 * CBMonthBegin(n=1, offset=timedelta(days=10)), {datetime(2021, 3, 1): datetime(2021, 1, 1) + timedelta(days=10), datetime(2021, 4, 3): datetime(2021, 3, 1) + timedelta(days=10)}), (CBMonthBegin(n=0, offset=timedelta(days=1)), {datetime(2021, 3, 2): datetime(2021, 4, 1) + timedelta(days=1), datetime(2021, 4, 1): datetime(2021, 4, 1) + timedelta(days=1)}), (CBMonthBegin(n=1, holidays=['2021-04-01', '2021-04-02'], offset=timedelta(days=1)), {datetime(2021, 3, 2): datetime(2021, 4, 5) + timedelta(days=1)})])
    def test_apply_with_extra_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)