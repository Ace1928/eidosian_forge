from datetime import datetime
from dateutil.relativedelta import relativedelta
import pytest
from pandas import Timestamp
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
from pandas.tseries.offsets import (
class TestFY5253LastOfMonthQuarter:

    def test_is_anchored(self):
        msg = 'FY5253Quarter.is_anchored is deprecated '
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert makeFY5253LastOfMonthQuarter(startingMonth=1, weekday=WeekDay.SAT, qtr_with_extra_week=4).is_anchored()
            assert makeFY5253LastOfMonthQuarter(weekday=WeekDay.SAT, startingMonth=3, qtr_with_extra_week=4).is_anchored()
            assert not makeFY5253LastOfMonthQuarter(2, startingMonth=1, weekday=WeekDay.SAT, qtr_with_extra_week=4).is_anchored()

    def test_equality(self):
        assert makeFY5253LastOfMonthQuarter(startingMonth=1, weekday=WeekDay.SAT, qtr_with_extra_week=4) == makeFY5253LastOfMonthQuarter(startingMonth=1, weekday=WeekDay.SAT, qtr_with_extra_week=4)
        assert makeFY5253LastOfMonthQuarter(startingMonth=1, weekday=WeekDay.SAT, qtr_with_extra_week=4) != makeFY5253LastOfMonthQuarter(startingMonth=1, weekday=WeekDay.SUN, qtr_with_extra_week=4)
        assert makeFY5253LastOfMonthQuarter(startingMonth=1, weekday=WeekDay.SAT, qtr_with_extra_week=4) != makeFY5253LastOfMonthQuarter(startingMonth=2, weekday=WeekDay.SAT, qtr_with_extra_week=4)

    def test_offset(self):
        offset = makeFY5253LastOfMonthQuarter(1, startingMonth=9, weekday=WeekDay.SAT, qtr_with_extra_week=4)
        offset2 = makeFY5253LastOfMonthQuarter(2, startingMonth=9, weekday=WeekDay.SAT, qtr_with_extra_week=4)
        offset4 = makeFY5253LastOfMonthQuarter(4, startingMonth=9, weekday=WeekDay.SAT, qtr_with_extra_week=4)
        offset_neg1 = makeFY5253LastOfMonthQuarter(-1, startingMonth=9, weekday=WeekDay.SAT, qtr_with_extra_week=4)
        offset_neg2 = makeFY5253LastOfMonthQuarter(-2, startingMonth=9, weekday=WeekDay.SAT, qtr_with_extra_week=4)
        GMCR = [datetime(2010, 3, 27), datetime(2010, 6, 26), datetime(2010, 9, 25), datetime(2010, 12, 25), datetime(2011, 3, 26), datetime(2011, 6, 25), datetime(2011, 9, 24), datetime(2011, 12, 24), datetime(2012, 3, 24), datetime(2012, 6, 23), datetime(2012, 9, 29), datetime(2012, 12, 29), datetime(2013, 3, 30), datetime(2013, 6, 29)]
        assert_offset_equal(offset, base=GMCR[0], expected=GMCR[1])
        assert_offset_equal(offset, base=GMCR[0] + relativedelta(days=-1), expected=GMCR[0])
        assert_offset_equal(offset, base=GMCR[1], expected=GMCR[2])
        assert_offset_equal(offset2, base=GMCR[0], expected=GMCR[2])
        assert_offset_equal(offset4, base=GMCR[0], expected=GMCR[4])
        assert_offset_equal(offset_neg1, base=GMCR[-1], expected=GMCR[-2])
        assert_offset_equal(offset_neg1, base=GMCR[-1] + relativedelta(days=+1), expected=GMCR[-1])
        assert_offset_equal(offset_neg2, base=GMCR[-1], expected=GMCR[-3])
        date = GMCR[0] + relativedelta(days=-1)
        for expected in GMCR:
            assert_offset_equal(offset, date, expected)
            date = date + offset
        date = GMCR[-1] + relativedelta(days=+1)
        for expected in reversed(GMCR):
            assert_offset_equal(offset_neg1, date, expected)
            date = date + offset_neg1
    lomq_aug_sat_4 = makeFY5253LastOfMonthQuarter(1, startingMonth=8, weekday=WeekDay.SAT, qtr_with_extra_week=4)
    lomq_sep_sat_4 = makeFY5253LastOfMonthQuarter(1, startingMonth=9, weekday=WeekDay.SAT, qtr_with_extra_week=4)
    on_offset_cases = [(lomq_aug_sat_4, datetime(2006, 8, 26), True), (lomq_aug_sat_4, datetime(2007, 8, 25), True), (lomq_aug_sat_4, datetime(2008, 8, 30), True), (lomq_aug_sat_4, datetime(2009, 8, 29), True), (lomq_aug_sat_4, datetime(2010, 8, 28), True), (lomq_aug_sat_4, datetime(2011, 8, 27), True), (lomq_aug_sat_4, datetime(2019, 8, 31), True), (lomq_aug_sat_4, datetime(2006, 8, 27), False), (lomq_aug_sat_4, datetime(2007, 8, 28), False), (lomq_aug_sat_4, datetime(2008, 8, 31), False), (lomq_aug_sat_4, datetime(2009, 8, 30), False), (lomq_aug_sat_4, datetime(2010, 8, 29), False), (lomq_aug_sat_4, datetime(2011, 8, 28), False), (lomq_aug_sat_4, datetime(2006, 8, 25), False), (lomq_aug_sat_4, datetime(2007, 8, 24), False), (lomq_aug_sat_4, datetime(2008, 8, 29), False), (lomq_aug_sat_4, datetime(2009, 8, 28), False), (lomq_aug_sat_4, datetime(2010, 8, 27), False), (lomq_aug_sat_4, datetime(2011, 8, 26), False), (lomq_aug_sat_4, datetime(2019, 8, 30), False), (lomq_sep_sat_4, datetime(2010, 9, 25), True), (lomq_sep_sat_4, datetime(2011, 9, 24), True), (lomq_sep_sat_4, datetime(2012, 9, 29), True), (lomq_sep_sat_4, datetime(2013, 6, 29), True), (lomq_sep_sat_4, datetime(2012, 6, 23), True), (lomq_sep_sat_4, datetime(2012, 6, 30), False), (lomq_sep_sat_4, datetime(2013, 3, 30), True), (lomq_sep_sat_4, datetime(2012, 3, 24), True), (lomq_sep_sat_4, datetime(2012, 12, 29), True), (lomq_sep_sat_4, datetime(2011, 12, 24), True), (makeFY5253LastOfMonthQuarter(1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1), datetime(2011, 4, 2), True), (makeFY5253LastOfMonthQuarter(1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1), datetime(2012, 12, 29), True), (makeFY5253LastOfMonthQuarter(1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1), datetime(2011, 12, 31), True), (makeFY5253LastOfMonthQuarter(1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1), datetime(2010, 12, 25), True)]

    @pytest.mark.parametrize('case', on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)

    def test_year_has_extra_week(self):
        assert makeFY5253LastOfMonthQuarter(1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1).year_has_extra_week(datetime(2011, 4, 2))
        assert makeFY5253LastOfMonthQuarter(1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1).year_has_extra_week(datetime(2010, 12, 26))
        assert not makeFY5253LastOfMonthQuarter(1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1).year_has_extra_week(datetime(2010, 12, 25))
        for year in [x for x in range(1994, 2011 + 1) if x not in [2011, 2005, 2000, 1994]]:
            assert not makeFY5253LastOfMonthQuarter(1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1).year_has_extra_week(datetime(year, 4, 2))
        assert makeFY5253LastOfMonthQuarter(1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1).year_has_extra_week(datetime(2005, 4, 2))
        assert makeFY5253LastOfMonthQuarter(1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1).year_has_extra_week(datetime(2000, 4, 2))
        assert makeFY5253LastOfMonthQuarter(1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1).year_has_extra_week(datetime(1994, 4, 2))

    def test_get_weeks(self):
        sat_dec_1 = makeFY5253LastOfMonthQuarter(1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=1)
        sat_dec_4 = makeFY5253LastOfMonthQuarter(1, startingMonth=12, weekday=WeekDay.SAT, qtr_with_extra_week=4)
        assert sat_dec_1.get_weeks(datetime(2011, 4, 2)) == [14, 13, 13, 13]
        assert sat_dec_4.get_weeks(datetime(2011, 4, 2)) == [13, 13, 13, 14]
        assert sat_dec_1.get_weeks(datetime(2010, 12, 25)) == [13, 13, 13, 13]