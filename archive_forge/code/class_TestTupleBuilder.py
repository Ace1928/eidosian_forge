import unittest
import aniso8601
from aniso8601.builders import (
from aniso8601.exceptions import (
from aniso8601.tests.compat import mock
class TestTupleBuilder(unittest.TestCase):

    def test_build_date(self):
        datetuple = TupleBuilder.build_date()
        self.assertEqual(datetuple, DateTuple(None, None, None, None, None, None))
        datetuple = TupleBuilder.build_date(YYYY='1', MM='2', DD='3', Www='4', D='5', DDD='6')
        self.assertEqual(datetuple, DateTuple('1', '2', '3', '4', '5', '6'))

    def test_build_time(self):
        testtuples = (({}, TimeTuple(None, None, None, None)), ({'hh': '1', 'mm': '2', 'ss': '3', 'tz': None}, TimeTuple('1', '2', '3', None)), ({'hh': '1', 'mm': '2', 'ss': '3', 'tz': TimezoneTuple(False, False, '4', '5', 'tz name')}, TimeTuple('1', '2', '3', TimezoneTuple(False, False, '4', '5', 'tz name'))))
        for testtuple in testtuples:
            self.assertEqual(TupleBuilder.build_time(**testtuple[0]), testtuple[1])

    def test_build_datetime(self):
        testtuples = (({'date': DateTuple('1', '2', '3', '4', '5', '6'), 'time': TimeTuple('7', '8', '9', None)}, DatetimeTuple(DateTuple('1', '2', '3', '4', '5', '6'), TimeTuple('7', '8', '9', None))), ({'date': DateTuple('1', '2', '3', '4', '5', '6'), 'time': TimeTuple('7', '8', '9', TimezoneTuple(True, False, '10', '11', 'tz name'))}, DatetimeTuple(DateTuple('1', '2', '3', '4', '5', '6'), TimeTuple('7', '8', '9', TimezoneTuple(True, False, '10', '11', 'tz name')))))
        for testtuple in testtuples:
            self.assertEqual(TupleBuilder.build_datetime(**testtuple[0]), testtuple[1])

    def test_build_duration(self):
        testtuples = (({}, DurationTuple(None, None, None, None, None, None, None)), ({'PnY': '1', 'PnM': '2', 'PnW': '3', 'PnD': '4', 'TnH': '5', 'TnM': '6', 'TnS': '7'}, DurationTuple('1', '2', '3', '4', '5', '6', '7')))
        for testtuple in testtuples:
            self.assertEqual(TupleBuilder.build_duration(**testtuple[0]), testtuple[1])

    def test_build_interval(self):
        testtuples = (({}, IntervalTuple(None, None, None)), ({'start': DateTuple('1', '2', '3', '4', '5', '6'), 'end': DateTuple('7', '8', '9', '10', '11', '12')}, IntervalTuple(DateTuple('1', '2', '3', '4', '5', '6'), DateTuple('7', '8', '9', '10', '11', '12'), None)), ({'start': TimeTuple('1', '2', '3', TimezoneTuple(True, False, '7', '8', 'tz name')), 'end': TimeTuple('4', '5', '6', TimezoneTuple(False, False, '9', '10', 'tz name'))}, IntervalTuple(TimeTuple('1', '2', '3', TimezoneTuple(True, False, '7', '8', 'tz name')), TimeTuple('4', '5', '6', TimezoneTuple(False, False, '9', '10', 'tz name')), None)), ({'start': DatetimeTuple(DateTuple('1', '2', '3', '4', '5', '6'), TimeTuple('7', '8', '9', TimezoneTuple(True, False, '10', '11', 'tz name'))), 'end': DatetimeTuple(DateTuple('12', '13', '14', '15', '16', '17'), TimeTuple('18', '19', '20', TimezoneTuple(False, False, '21', '22', 'tz name')))}, IntervalTuple(DatetimeTuple(DateTuple('1', '2', '3', '4', '5', '6'), TimeTuple('7', '8', '9', TimezoneTuple(True, False, '10', '11', 'tz name'))), DatetimeTuple(DateTuple('12', '13', '14', '15', '16', '17'), TimeTuple('18', '19', '20', TimezoneTuple(False, False, '21', '22', 'tz name'))), None)), ({'start': DateTuple('1', '2', '3', '4', '5', '6'), 'end': None, 'duration': DurationTuple('7', '8', '9', '10', '11', '12', '13')}, IntervalTuple(DateTuple('1', '2', '3', '4', '5', '6'), None, DurationTuple('7', '8', '9', '10', '11', '12', '13'))), ({'start': None, 'end': TimeTuple('1', '2', '3', TimezoneTuple(True, False, '4', '5', 'tz name')), 'duration': DurationTuple('6', '7', '8', '9', '10', '11', '12')}, IntervalTuple(None, TimeTuple('1', '2', '3', TimezoneTuple(True, False, '4', '5', 'tz name')), DurationTuple('6', '7', '8', '9', '10', '11', '12'))))
        for testtuple in testtuples:
            self.assertEqual(TupleBuilder.build_interval(**testtuple[0]), testtuple[1])

    def test_build_repeating_interval(self):
        testtuples = (({}, RepeatingIntervalTuple(None, None, None)), ({'R': True, 'interval': IntervalTuple(DateTuple('1', '2', '3', '4', '5', '6'), DateTuple('7', '8', '9', '10', '11', '12'), None)}, RepeatingIntervalTuple(True, None, IntervalTuple(DateTuple('1', '2', '3', '4', '5', '6'), DateTuple('7', '8', '9', '10', '11', '12'), None))), ({'R': False, 'Rnn': '1', 'interval': IntervalTuple(DatetimeTuple(DateTuple('2', '3', '4', '5', '6', '7'), TimeTuple('8', '9', '10', None)), DatetimeTuple(DateTuple('11', '12', '13', '14', '15', '16'), TimeTuple('17', '18', '19', None)), None)}, RepeatingIntervalTuple(False, '1', IntervalTuple(DatetimeTuple(DateTuple('2', '3', '4', '5', '6', '7'), TimeTuple('8', '9', '10', None)), DatetimeTuple(DateTuple('11', '12', '13', '14', '15', '16'), TimeTuple('17', '18', '19', None)), None))))
        for testtuple in testtuples:
            result = TupleBuilder.build_repeating_interval(**testtuple[0])
            self.assertEqual(result, testtuple[1])

    def test_build_timezone(self):
        testtuples = (({}, TimezoneTuple(None, None, None, None, '')), ({'negative': False, 'Z': True, 'name': 'UTC'}, TimezoneTuple(False, True, None, None, 'UTC')), ({'negative': False, 'Z': False, 'hh': '1', 'mm': '2', 'name': '+01:02'}, TimezoneTuple(False, False, '1', '2', '+01:02')), ({'negative': True, 'Z': False, 'hh': '1', 'mm': '2', 'name': '-01:02'}, TimezoneTuple(True, False, '1', '2', '-01:02')))
        for testtuple in testtuples:
            result = TupleBuilder.build_timezone(**testtuple[0])
            self.assertEqual(result, testtuple[1])