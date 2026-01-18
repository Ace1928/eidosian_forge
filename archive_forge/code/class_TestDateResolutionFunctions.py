import unittest
import aniso8601
from aniso8601.date import get_date_resolution, parse_date
from aniso8601.exceptions import DayOutOfBoundsError, ISOFormatError
from aniso8601.resolution import DateResolution
from aniso8601.tests.compat import mock
class TestDateResolutionFunctions(unittest.TestCase):

    def test_get_date_resolution_year(self):
        self.assertEqual(get_date_resolution('2013'), DateResolution.Year)
        self.assertEqual(get_date_resolution('0001'), DateResolution.Year)
        self.assertEqual(get_date_resolution('19'), DateResolution.Year)

    def test_get_date_resolution_month(self):
        self.assertEqual(get_date_resolution('1981-04'), DateResolution.Month)

    def test_get_date_resolution_week(self):
        self.assertEqual(get_date_resolution('2004-W53'), DateResolution.Week)
        self.assertEqual(get_date_resolution('2009-W01'), DateResolution.Week)
        self.assertEqual(get_date_resolution('2004W53'), DateResolution.Week)

    def test_get_date_resolution_day(self):
        self.assertEqual(get_date_resolution('2004-04-11'), DateResolution.Day)
        self.assertEqual(get_date_resolution('20090121'), DateResolution.Day)

    def test_get_date_resolution_year_weekday(self):
        self.assertEqual(get_date_resolution('2004-W53-6'), DateResolution.Weekday)
        self.assertEqual(get_date_resolution('2004W536'), DateResolution.Weekday)

    def test_get_date_resolution_year_ordinal(self):
        self.assertEqual(get_date_resolution('1981-095'), DateResolution.Ordinal)
        self.assertEqual(get_date_resolution('1981095'), DateResolution.Ordinal)

    def test_get_date_resolution_badtype(self):
        testtuples = (None, 1, False, 1.234)
        for testtuple in testtuples:
            with self.assertRaises(ValueError):
                get_date_resolution(testtuple)

    def test_get_date_resolution_extended_year(self):
        testtuples = ('+2000', '+30000')
        for testtuple in testtuples:
            with self.assertRaises(NotImplementedError):
                get_date_resolution(testtuple)

    def test_get_date_resolution_badweek(self):
        testtuples = ('2004-W1', '2004W1')
        for testtuple in testtuples:
            with self.assertRaises(ISOFormatError):
                get_date_resolution(testtuple)

    def test_get_date_resolution_badweekday(self):
        testtuples = ('2004-W53-67', '2004W5367')
        for testtuple in testtuples:
            with self.assertRaises(ISOFormatError):
                get_date_resolution(testtuple)

    def test_get_date_resolution_badstr(self):
        testtuples = ('W53', '2004-W', '2014-01-230', '2014-012-23', '201-01-23', '201401230', '201401', '')
        for testtuple in testtuples:
            with self.assertRaises(ISOFormatError):
                get_date_resolution(testtuple)