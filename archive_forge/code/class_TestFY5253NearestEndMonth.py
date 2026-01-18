from datetime import datetime
from dateutil.relativedelta import relativedelta
import pytest
from pandas import Timestamp
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
from pandas.tseries.offsets import (
class TestFY5253NearestEndMonth:

    def test_get_year_end(self):
        assert makeFY5253NearestEndMonth(startingMonth=8, weekday=WeekDay.SAT).get_year_end(datetime(2013, 1, 1)) == datetime(2013, 8, 31)
        assert makeFY5253NearestEndMonth(startingMonth=8, weekday=WeekDay.SUN).get_year_end(datetime(2013, 1, 1)) == datetime(2013, 9, 1)
        assert makeFY5253NearestEndMonth(startingMonth=8, weekday=WeekDay.FRI).get_year_end(datetime(2013, 1, 1)) == datetime(2013, 8, 30)
        offset_n = FY5253(weekday=WeekDay.TUE, startingMonth=12, variation='nearest')
        assert offset_n.get_year_end(datetime(2012, 1, 1)) == datetime(2013, 1, 1)
        assert offset_n.get_year_end(datetime(2012, 1, 10)) == datetime(2013, 1, 1)
        assert offset_n.get_year_end(datetime(2013, 1, 1)) == datetime(2013, 12, 31)
        assert offset_n.get_year_end(datetime(2013, 1, 2)) == datetime(2013, 12, 31)
        assert offset_n.get_year_end(datetime(2013, 1, 3)) == datetime(2013, 12, 31)
        assert offset_n.get_year_end(datetime(2013, 1, 10)) == datetime(2013, 12, 31)
        JNJ = FY5253(n=1, startingMonth=12, weekday=6, variation='nearest')
        assert JNJ.get_year_end(datetime(2006, 1, 1)) == datetime(2006, 12, 31)
    offset_lom_aug_sat = makeFY5253NearestEndMonth(1, startingMonth=8, weekday=WeekDay.SAT)
    offset_lom_aug_thu = makeFY5253NearestEndMonth(1, startingMonth=8, weekday=WeekDay.THU)
    offset_n = FY5253(weekday=WeekDay.TUE, startingMonth=12, variation='nearest')
    on_offset_cases = [(offset_lom_aug_sat, datetime(2006, 9, 2), True), (offset_lom_aug_sat, datetime(2007, 9, 1), True), (offset_lom_aug_sat, datetime(2008, 8, 30), True), (offset_lom_aug_sat, datetime(2009, 8, 29), True), (offset_lom_aug_sat, datetime(2010, 8, 28), True), (offset_lom_aug_sat, datetime(2011, 9, 3), True), (offset_lom_aug_sat, datetime(2016, 9, 3), True), (offset_lom_aug_sat, datetime(2017, 9, 2), True), (offset_lom_aug_sat, datetime(2018, 9, 1), True), (offset_lom_aug_sat, datetime(2019, 8, 31), True), (offset_lom_aug_sat, datetime(2006, 8, 27), False), (offset_lom_aug_sat, datetime(2007, 8, 28), False), (offset_lom_aug_sat, datetime(2008, 8, 31), False), (offset_lom_aug_sat, datetime(2009, 8, 30), False), (offset_lom_aug_sat, datetime(2010, 8, 29), False), (offset_lom_aug_sat, datetime(2011, 8, 28), False), (offset_lom_aug_sat, datetime(2006, 8, 25), False), (offset_lom_aug_sat, datetime(2007, 8, 24), False), (offset_lom_aug_sat, datetime(2008, 8, 29), False), (offset_lom_aug_sat, datetime(2009, 8, 28), False), (offset_lom_aug_sat, datetime(2010, 8, 27), False), (offset_lom_aug_sat, datetime(2011, 8, 26), False), (offset_lom_aug_sat, datetime(2019, 8, 30), False), (offset_lom_aug_thu, datetime(2012, 8, 30), True), (offset_lom_aug_thu, datetime(2011, 9, 1), True), (offset_n, datetime(2012, 12, 31), False), (offset_n, datetime(2013, 1, 1), True), (offset_n, datetime(2013, 1, 2), False)]

    @pytest.mark.parametrize('case', on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)

    def test_apply(self):
        date_seq_nem_8_sat = [datetime(2006, 9, 2), datetime(2007, 9, 1), datetime(2008, 8, 30), datetime(2009, 8, 29), datetime(2010, 8, 28), datetime(2011, 9, 3)]
        JNJ = [datetime(2005, 1, 2), datetime(2006, 1, 1), datetime(2006, 12, 31), datetime(2007, 12, 30), datetime(2008, 12, 28), datetime(2010, 1, 3), datetime(2011, 1, 2), datetime(2012, 1, 1), datetime(2012, 12, 30)]
        DEC_SAT = FY5253(n=-1, startingMonth=12, weekday=5, variation='nearest')
        tests = [(makeFY5253NearestEndMonth(startingMonth=8, weekday=WeekDay.SAT), date_seq_nem_8_sat), (makeFY5253NearestEndMonth(n=1, startingMonth=8, weekday=WeekDay.SAT), date_seq_nem_8_sat), (makeFY5253NearestEndMonth(startingMonth=8, weekday=WeekDay.SAT), [datetime(2006, 9, 1)] + date_seq_nem_8_sat), (makeFY5253NearestEndMonth(n=1, startingMonth=8, weekday=WeekDay.SAT), [datetime(2006, 9, 3)] + date_seq_nem_8_sat[1:]), (makeFY5253NearestEndMonth(n=-1, startingMonth=8, weekday=WeekDay.SAT), list(reversed(date_seq_nem_8_sat))), (makeFY5253NearestEndMonth(n=1, startingMonth=12, weekday=WeekDay.SUN), JNJ), (makeFY5253NearestEndMonth(n=-1, startingMonth=12, weekday=WeekDay.SUN), list(reversed(JNJ))), (makeFY5253NearestEndMonth(n=1, startingMonth=12, weekday=WeekDay.SUN), [datetime(2005, 1, 2), datetime(2006, 1, 1)]), (makeFY5253NearestEndMonth(n=1, startingMonth=12, weekday=WeekDay.SUN), [datetime(2006, 1, 2), datetime(2006, 12, 31)]), (DEC_SAT, [datetime(2013, 1, 15), datetime(2012, 12, 29)])]
        for test in tests:
            offset, data = test
            current = data[0]
            for datum in data[1:]:
                current = current + offset
                assert current == datum