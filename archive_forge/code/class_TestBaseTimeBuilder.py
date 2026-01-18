import unittest
import aniso8601
from aniso8601.builders import (
from aniso8601.exceptions import (
from aniso8601.tests.compat import mock
class TestBaseTimeBuilder(unittest.TestCase):

    def test_build_date(self):
        with self.assertRaises(NotImplementedError):
            BaseTimeBuilder.build_date()

    def test_build_time(self):
        with self.assertRaises(NotImplementedError):
            BaseTimeBuilder.build_time()

    def test_build_datetime(self):
        with self.assertRaises(NotImplementedError):
            BaseTimeBuilder.build_datetime(None, None)

    def test_build_duration(self):
        with self.assertRaises(NotImplementedError):
            BaseTimeBuilder.build_duration()

    def test_build_interval(self):
        with self.assertRaises(NotImplementedError):
            BaseTimeBuilder.build_interval()

    def test_build_repeating_interval(self):
        with self.assertRaises(NotImplementedError):
            BaseTimeBuilder.build_repeating_interval()

    def test_build_timezone(self):
        with self.assertRaises(NotImplementedError):
            BaseTimeBuilder.build_timezone()

    def test_range_check_date(self):
        with self.assertRaises(DayOutOfBoundsError):
            BaseTimeBuilder.range_check_date(YYYY='0007', MM='02', DD='30')
        with self.assertRaises(DayOutOfBoundsError):
            BaseTimeBuilder.range_check_date(YYYY='0007', DDD='366')
        with self.assertRaises(MonthOutOfBoundsError):
            BaseTimeBuilder.range_check_date(YYYY='4333', MM='30', DD='30')
        with self.assertRaises(WeekOutOfBoundsError):
            BaseTimeBuilder.range_check_date(YYYY='2003', Www='00')
        with self.assertRaises(WeekOutOfBoundsError):
            BaseTimeBuilder.range_check_date(YYYY='2004', Www='54')
        with self.assertRaises(DayOutOfBoundsError):
            BaseTimeBuilder.range_check_date(YYYY='2001', Www='02', D='0')
        with self.assertRaises(DayOutOfBoundsError):
            BaseTimeBuilder.range_check_date(YYYY='2001', Www='02', D='8')
        with self.assertRaises(DayOutOfBoundsError):
            BaseTimeBuilder.range_check_date(YYYY='1981', DDD='000')
        with self.assertRaises(DayOutOfBoundsError):
            BaseTimeBuilder.range_check_date(YYYY='1234', DDD='000')
        with self.assertRaises(DayOutOfBoundsError):
            BaseTimeBuilder.range_check_date(YYYY='1234', DDD='367')
        with self.assertRaises(DayOutOfBoundsError):
            BaseTimeBuilder.range_check_date(YYYY='1981', DDD='366')
        self.assertEqual(BaseTimeBuilder.range_check_date(rangedict={}), (None, None, None, None, None, None))

    def test_range_check_time(self):
        with self.assertRaises(LeapSecondError):
            BaseTimeBuilder.range_check_time(hh='23', mm='59', ss='60')
        with self.assertRaises(SecondsOutOfBoundsError):
            BaseTimeBuilder.range_check_time(hh='00', mm='00', ss='60')
        with self.assertRaises(SecondsOutOfBoundsError):
            BaseTimeBuilder.range_check_time(hh='00', mm='00', ss='61')
        with self.assertRaises(MinutesOutOfBoundsError):
            BaseTimeBuilder.range_check_time(hh='00', mm='61')
        with self.assertRaises(MinutesOutOfBoundsError):
            BaseTimeBuilder.range_check_time(hh='00', mm='60')
        with self.assertRaises(MinutesOutOfBoundsError):
            BaseTimeBuilder.range_check_time(hh='00', mm='60.1')
        with self.assertRaises(HoursOutOfBoundsError):
            BaseTimeBuilder.range_check_time(hh='25')
        with self.assertRaises(MidnightBoundsError):
            BaseTimeBuilder.range_check_time(hh='24', mm='00', ss='01')
        with self.assertRaises(MidnightBoundsError):
            BaseTimeBuilder.range_check_time(hh='24', mm='00.1')
        with self.assertRaises(MidnightBoundsError):
            BaseTimeBuilder.range_check_time(hh='24', mm='01')
        with self.assertRaises(MidnightBoundsError):
            BaseTimeBuilder.range_check_time(hh='24.1')
        with self.assertRaises(LeapSecondError):
            BaseTimeBuilder.range_check_time(hh='23', mm='59', ss='60')
        self.assertEqual(BaseTimeBuilder.range_check_time(rangedict={}), (None, None, None, None))

    def test_range_check_time_leap_seconds_supported(self):
        self.assertEqual(LeapSecondSupportingTestBuilder.range_check_time(hh='23', mm='59', ss='60'), (23, 59, 60, None))
        with self.assertRaises(SecondsOutOfBoundsError):
            LeapSecondSupportingTestBuilder.range_check_time(hh='01', mm='02', ss='60')

    def test_range_check_duration(self):
        self.assertEqual(BaseTimeBuilder.range_check_duration(), (None, None, None, None, None, None, None))
        self.assertEqual(BaseTimeBuilder.range_check_duration(rangedict={}), (None, None, None, None, None, None, None))

    def test_range_check_repeating_interval(self):
        self.assertEqual(BaseTimeBuilder.range_check_repeating_interval(), (None, None, None))
        self.assertEqual(BaseTimeBuilder.range_check_repeating_interval(rangedict={}), (None, None, None))

    def test_range_check_timezone(self):
        self.assertEqual(BaseTimeBuilder.range_check_timezone(), (None, None, None, None, ''))
        self.assertEqual(BaseTimeBuilder.range_check_timezone(rangedict={}), (None, None, None, None, ''))

    def test_build_object(self):
        datetest = (DateTuple('1', '2', '3', '4', '5', '6'), {'YYYY': '1', 'MM': '2', 'DD': '3', 'Www': '4', 'D': '5', 'DDD': '6'})
        timetest = (TimeTuple('1', '2', '3', TimezoneTuple(False, False, '4', '5', 'tz name')), {'hh': '1', 'mm': '2', 'ss': '3', 'tz': TimezoneTuple(False, False, '4', '5', 'tz name')})
        datetimetest = (DatetimeTuple(DateTuple('1', '2', '3', '4', '5', '6'), TimeTuple('7', '8', '9', TimezoneTuple(True, False, '10', '11', 'tz name'))), (DateTuple('1', '2', '3', '4', '5', '6'), TimeTuple('7', '8', '9', TimezoneTuple(True, False, '10', '11', 'tz name'))))
        durationtest = (DurationTuple('1', '2', '3', '4', '5', '6', '7'), {'PnY': '1', 'PnM': '2', 'PnW': '3', 'PnD': '4', 'TnH': '5', 'TnM': '6', 'TnS': '7'})
        intervaltests = ((IntervalTuple(DateTuple('1', '2', '3', '4', '5', '6'), DateTuple('7', '8', '9', '10', '11', '12'), None), {'start': DateTuple('1', '2', '3', '4', '5', '6'), 'end': DateTuple('7', '8', '9', '10', '11', '12'), 'duration': None}), (IntervalTuple(DateTuple('1', '2', '3', '4', '5', '6'), None, DurationTuple('7', '8', '9', '10', '11', '12', '13')), {'start': DateTuple('1', '2', '3', '4', '5', '6'), 'end': None, 'duration': DurationTuple('7', '8', '9', '10', '11', '12', '13')}), (IntervalTuple(None, TimeTuple('1', '2', '3', TimezoneTuple(True, False, '4', '5', 'tz name')), DurationTuple('6', '7', '8', '9', '10', '11', '12')), {'start': None, 'end': TimeTuple('1', '2', '3', TimezoneTuple(True, False, '4', '5', 'tz name')), 'duration': DurationTuple('6', '7', '8', '9', '10', '11', '12')}))
        repeatingintervaltests = ((RepeatingIntervalTuple(True, None, IntervalTuple(DateTuple('1', '2', '3', '4', '5', '6'), DateTuple('7', '8', '9', '10', '11', '12'), None)), {'R': True, 'Rnn': None, 'interval': IntervalTuple(DateTuple('1', '2', '3', '4', '5', '6'), DateTuple('7', '8', '9', '10', '11', '12'), None)}), (RepeatingIntervalTuple(False, '1', IntervalTuple(DatetimeTuple(DateTuple('2', '3', '4', '5', '6', '7'), TimeTuple('8', '9', '10', None)), DatetimeTuple(DateTuple('11', '12', '13', '14', '15', '16'), TimeTuple('17', '18', '19', None)), None)), {'R': False, 'Rnn': '1', 'interval': IntervalTuple(DatetimeTuple(DateTuple('2', '3', '4', '5', '6', '7'), TimeTuple('8', '9', '10', None)), DatetimeTuple(DateTuple('11', '12', '13', '14', '15', '16'), TimeTuple('17', '18', '19', None)), None)}))
        timezonetest = (TimezoneTuple(False, False, '1', '2', '+01:02'), {'negative': False, 'Z': False, 'hh': '1', 'mm': '2', 'name': '+01:02'})
        with mock.patch.object(aniso8601.builders.BaseTimeBuilder, 'build_date') as mock_build:
            mock_build.return_value = datetest[0]
            result = BaseTimeBuilder._build_object(datetest[0])
            self.assertEqual(result, datetest[0])
            mock_build.assert_called_once_with(**datetest[1])
        with mock.patch.object(aniso8601.builders.BaseTimeBuilder, 'build_time') as mock_build:
            mock_build.return_value = timetest[0]
            result = BaseTimeBuilder._build_object(timetest[0])
            self.assertEqual(result, timetest[0])
            mock_build.assert_called_once_with(**timetest[1])
        with mock.patch.object(aniso8601.builders.BaseTimeBuilder, 'build_datetime') as mock_build:
            mock_build.return_value = datetimetest[0]
            result = BaseTimeBuilder._build_object(datetimetest[0])
            self.assertEqual(result, datetimetest[0])
            mock_build.assert_called_once_with(*datetimetest[1])
        with mock.patch.object(aniso8601.builders.BaseTimeBuilder, 'build_duration') as mock_build:
            mock_build.return_value = durationtest[0]
            result = BaseTimeBuilder._build_object(durationtest[0])
            self.assertEqual(result, durationtest[0])
            mock_build.assert_called_once_with(**durationtest[1])
        for intervaltest in intervaltests:
            with mock.patch.object(aniso8601.builders.BaseTimeBuilder, 'build_interval') as mock_build:
                mock_build.return_value = intervaltest[0]
                result = BaseTimeBuilder._build_object(intervaltest[0])
                self.assertEqual(result, intervaltest[0])
                mock_build.assert_called_once_with(**intervaltest[1])
        for repeatingintervaltest in repeatingintervaltests:
            with mock.patch.object(aniso8601.builders.BaseTimeBuilder, 'build_repeating_interval') as mock_build:
                mock_build.return_value = repeatingintervaltest[0]
                result = BaseTimeBuilder._build_object(repeatingintervaltest[0])
                self.assertEqual(result, repeatingintervaltest[0])
                mock_build.assert_called_once_with(**repeatingintervaltest[1])
        with mock.patch.object(aniso8601.builders.BaseTimeBuilder, 'build_timezone') as mock_build:
            mock_build.return_value = timezonetest[0]
            result = BaseTimeBuilder._build_object(timezonetest[0])
            self.assertEqual(result, timezonetest[0])
            mock_build.assert_called_once_with(**timezonetest[1])

    def test_is_interval_end_concise(self):
        self.assertTrue(BaseTimeBuilder._is_interval_end_concise(TimeTuple('1', '2', '3', None)))
        self.assertTrue(BaseTimeBuilder._is_interval_end_concise(DateTuple(None, '2', '3', '4', '5', '6')))
        self.assertTrue(BaseTimeBuilder._is_interval_end_concise(DatetimeTuple(DateTuple(None, '2', '3', '4', '5', '6'), TimeTuple('7', '8', '9', None))))
        self.assertFalse(BaseTimeBuilder._is_interval_end_concise(DateTuple('1', '2', '3', '4', '5', '6')))
        self.assertFalse(BaseTimeBuilder._is_interval_end_concise(DatetimeTuple(DateTuple('1', '2', '3', '4', '5', '6'), TimeTuple('7', '8', '9', None))))

    def test_combine_concise_interval_tuples(self):
        testtuples = ((DateTuple('2020', '01', '01', None, None, None), DateTuple(None, None, '02', None, None, None), DateTuple('2020', '01', '02', None, None, None)), (DateTuple('2008', '02', '15', None, None, None), DateTuple(None, '03', '14', None, None, None), DateTuple('2008', '03', '14', None, None, None)), (DatetimeTuple(DateTuple('2007', '12', '14', None, None, None), TimeTuple('13', '30', None, None)), TimeTuple('15', '30', None, None), DatetimeTuple(DateTuple('2007', '12', '14', None, None, None), TimeTuple('15', '30', None, None))), (DatetimeTuple(DateTuple('2007', '11', '13', None, None, None), TimeTuple('09', '00', None, None)), DatetimeTuple(DateTuple(None, None, '15', None, None, None), TimeTuple('17', '00', None, None)), DatetimeTuple(DateTuple('2007', '11', '15', None, None, None), TimeTuple('17', '00', None, None))), (DatetimeTuple(DateTuple('2007', '11', '13', None, None, None), TimeTuple('00', '00', None, None)), DatetimeTuple(DateTuple(None, None, '16', None, None, None), TimeTuple('00', '00', None, None)), DatetimeTuple(DateTuple('2007', '11', '16', None, None, None), TimeTuple('00', '00', None, None))), (DatetimeTuple(DateTuple('2007', '11', '13', None, None, None), TimeTuple('09', '00', None, TimezoneTuple(False, True, None, None, 'Z'))), DatetimeTuple(DateTuple(None, None, '15', None, None, None), TimeTuple('17', '00', None, None)), DatetimeTuple(DateTuple('2007', '11', '15', None, None, None), TimeTuple('17', '00', None, TimezoneTuple(False, True, None, None, 'Z')))))
        for testtuple in testtuples:
            result = BaseTimeBuilder._combine_concise_interval_tuples(testtuple[0], testtuple[1])
            self.assertEqual(result, testtuple[2])