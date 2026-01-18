import unittest
from datetime import datetime
from .dateutil import UTC, TimezoneInfo, format_rfc3339, parse_rfc3339
def _parse_rfc3339_test(self, st, y, m, d, h, mn, s):
    actual = parse_rfc3339(st)
    expected = datetime(y, m, d, h, mn, s, 0, UTC)
    self.assertEqual(expected, actual)