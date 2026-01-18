from datetime import datetime
from dateutil.relativedelta import relativedelta
import pytest
from pandas import Timestamp
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
from pandas.tseries.offsets import (
class TestFY5253LastOfMonth:
    offset_lom_sat_aug = makeFY5253LastOfMonth(1, startingMonth=8, weekday=WeekDay.SAT)
    offset_lom_sat_sep = makeFY5253LastOfMonth(1, startingMonth=9, weekday=WeekDay.SAT)
    on_offset_cases = [(offset_lom_sat_aug, datetime(2006, 8, 26), True), (offset_lom_sat_aug, datetime(2007, 8, 25), True), (offset_lom_sat_aug, datetime(2008, 8, 30), True), (offset_lom_sat_aug, datetime(2009, 8, 29), True), (offset_lom_sat_aug, datetime(2010, 8, 28), True), (offset_lom_sat_aug, datetime(2011, 8, 27), True), (offset_lom_sat_aug, datetime(2012, 8, 25), True), (offset_lom_sat_aug, datetime(2013, 8, 31), True), (offset_lom_sat_aug, datetime(2014, 8, 30), True), (offset_lom_sat_aug, datetime(2015, 8, 29), True), (offset_lom_sat_aug, datetime(2016, 8, 27), True), (offset_lom_sat_aug, datetime(2017, 8, 26), True), (offset_lom_sat_aug, datetime(2018, 8, 25), True), (offset_lom_sat_aug, datetime(2019, 8, 31), True), (offset_lom_sat_aug, datetime(2006, 8, 27), False), (offset_lom_sat_aug, datetime(2007, 8, 28), False), (offset_lom_sat_aug, datetime(2008, 8, 31), False), (offset_lom_sat_aug, datetime(2009, 8, 30), False), (offset_lom_sat_aug, datetime(2010, 8, 29), False), (offset_lom_sat_aug, datetime(2011, 8, 28), False), (offset_lom_sat_aug, datetime(2006, 8, 25), False), (offset_lom_sat_aug, datetime(2007, 8, 24), False), (offset_lom_sat_aug, datetime(2008, 8, 29), False), (offset_lom_sat_aug, datetime(2009, 8, 28), False), (offset_lom_sat_aug, datetime(2010, 8, 27), False), (offset_lom_sat_aug, datetime(2011, 8, 26), False), (offset_lom_sat_aug, datetime(2019, 8, 30), False), (offset_lom_sat_sep, datetime(2010, 9, 25), True), (offset_lom_sat_sep, datetime(2011, 9, 24), True), (offset_lom_sat_sep, datetime(2012, 9, 29), True)]

    @pytest.mark.parametrize('case', on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)

    def test_apply(self):
        offset_lom_aug_sat = makeFY5253LastOfMonth(startingMonth=8, weekday=WeekDay.SAT)
        offset_lom_aug_sat_1 = makeFY5253LastOfMonth(n=1, startingMonth=8, weekday=WeekDay.SAT)
        date_seq_lom_aug_sat = [datetime(2006, 8, 26), datetime(2007, 8, 25), datetime(2008, 8, 30), datetime(2009, 8, 29), datetime(2010, 8, 28), datetime(2011, 8, 27), datetime(2012, 8, 25), datetime(2013, 8, 31), datetime(2014, 8, 30), datetime(2015, 8, 29), datetime(2016, 8, 27)]
        tests = [(offset_lom_aug_sat, date_seq_lom_aug_sat), (offset_lom_aug_sat_1, date_seq_lom_aug_sat), (offset_lom_aug_sat, [datetime(2006, 8, 25)] + date_seq_lom_aug_sat), (offset_lom_aug_sat_1, [datetime(2006, 8, 27)] + date_seq_lom_aug_sat[1:]), (makeFY5253LastOfMonth(n=-1, startingMonth=8, weekday=WeekDay.SAT), list(reversed(date_seq_lom_aug_sat)))]
        for test in tests:
            offset, data = test
            current = data[0]
            for datum in data[1:]:
                current = current + offset
                assert current == datum