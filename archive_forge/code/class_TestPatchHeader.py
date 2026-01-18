import random
import time
from breezy import tests, timestamp
from breezy.osutils import local_time_offset
class TestPatchHeader(tests.TestCase):

    def test_format_patch_date(self):
        self.assertEqual('1970-01-01 00:00:00 +0000', timestamp.format_patch_date(0))
        self.assertEqual('1970-01-01 00:00:00 +0000', timestamp.format_patch_date(0, 5 * 3600))
        self.assertEqual('1970-01-01 00:00:00 +0000', timestamp.format_patch_date(0, -5 * 3600))
        self.assertEqual('2007-03-06 10:04:19 -0500', timestamp.format_patch_date(1173193459, -5 * 3600))
        self.assertEqual('2007-03-06 09:34:19 -0530', timestamp.format_patch_date(1173193459, -5.5 * 3600))
        self.assertEqual('2007-03-06 15:05:19 +0001', timestamp.format_patch_date(1173193459, +1 * 60))

    def test_parse_patch_date(self):
        self.assertEqual((0, 0), timestamp.parse_patch_date('1970-01-01 00:00:00 +0000'))
        self.assertEqual((0, -5 * 3600), timestamp.parse_patch_date('1969-12-31 19:00:00 -0500'))
        self.assertEqual((0, +5 * 3600), timestamp.parse_patch_date('1970-01-01 05:00:00 +0500'))
        self.assertEqual((1173193459, -5 * 3600), timestamp.parse_patch_date('2007-03-06 10:04:19 -0500'))
        self.assertEqual((1173193459, +3 * 60), timestamp.parse_patch_date('2007-03-06 15:07:19 +0003'))
        self.assertEqual((1173193459, -5 * 3600), timestamp.parse_patch_date('2007-03-06 10:04:19-0500'))
        self.assertEqual((1173193459, -5 * 3600), timestamp.parse_patch_date('2007-03-06     10:04:19     -0500'))

    def test_parse_patch_date_bad(self):
        self.assertRaises(ValueError, timestamp.parse_patch_date, 'NOT A TIME')
        self.assertRaises(ValueError, timestamp.parse_patch_date, '2007-03-06 10:04:19 -0500x')
        self.assertRaises(ValueError, timestamp.parse_patch_date, '2007-03 10:04:19 -0500')
        self.assertRaises(ValueError, timestamp.parse_patch_date, '2007-03-06 10:04 -0500')
        self.assertRaises(ValueError, timestamp.parse_patch_date, '2007-03-06 10:04:19')
        self.assertRaises(ValueError, timestamp.parse_patch_date, '2007-03-06 10:04:19 0500')
        self.assertRaises(ValueError, timestamp.parse_patch_date, '2007-03-06 10:04:19 +2400')
        self.assertRaises(ValueError, timestamp.parse_patch_date, '2007-03-06 10:04:19 -2400')
        self.assertRaises(ValueError, timestamp.parse_patch_date, '2007-03-06 10:04:19 -0560')
        self.assertRaises(ValueError, timestamp.parse_patch_date, '2007-03-06 10:04:19 79500')
        self.assertRaises(ValueError, timestamp.parse_patch_date, '2007-03-06 10:04:19 +05-5')