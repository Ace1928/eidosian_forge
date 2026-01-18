import unittest
from datetime import datetime
from .dateutil import UTC, TimezoneInfo, format_rfc3339, parse_rfc3339
class DateUtilTest(unittest.TestCase):

    def _parse_rfc3339_test(self, st, y, m, d, h, mn, s):
        actual = parse_rfc3339(st)
        expected = datetime(y, m, d, h, mn, s, 0, UTC)
        self.assertEqual(expected, actual)

    def test_parse_rfc3339(self):
        self._parse_rfc3339_test('2017-07-25T04:44:21Z', 2017, 7, 25, 4, 44, 21)
        self._parse_rfc3339_test('2017-07-25 04:44:21Z', 2017, 7, 25, 4, 44, 21)
        self._parse_rfc3339_test('2017-07-25T04:44:21', 2017, 7, 25, 4, 44, 21)
        self._parse_rfc3339_test('2017-07-25T04:44:21z', 2017, 7, 25, 4, 44, 21)
        self._parse_rfc3339_test('2017-07-25T04:44:21+03:00', 2017, 7, 25, 1, 44, 21)
        self._parse_rfc3339_test('2017-07-25T04:44:21-03:00', 2017, 7, 25, 7, 44, 21)

    def test_format_rfc3339(self):
        self.assertEqual(format_rfc3339(datetime(2017, 7, 25, 4, 44, 21, 0, UTC)), '2017-07-25T04:44:21Z')
        self.assertEqual(format_rfc3339(datetime(2017, 7, 25, 4, 44, 21, 0, TimezoneInfo(2, 0))), '2017-07-25T02:44:21Z')
        self.assertEqual(format_rfc3339(datetime(2017, 7, 25, 4, 44, 21, 0, TimezoneInfo(-2, 30))), '2017-07-25T07:14:21Z')