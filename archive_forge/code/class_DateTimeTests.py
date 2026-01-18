import datetime
import sys
import types
import unittest
import six
from apitools.base.protorpclite import test_util
from apitools.base.protorpclite import util
class DateTimeTests(test_util.TestCase):

    def testDecodeDateTime(self):
        """Test that a RFC 3339 datetime string is decoded properly."""
        for datetime_string, datetime_vals in (('2012-09-30T15:31:50.262', (2012, 9, 30, 15, 31, 50, 262000)), ('2012-09-30T15:31:50', (2012, 9, 30, 15, 31, 50, 0))):
            decoded = util.decode_datetime(datetime_string)
            expected = datetime.datetime(*datetime_vals)
            self.assertEquals(expected, decoded)

    def testDecodeDateTimeWithTruncateTime(self):
        """Test that nanosec time is truncated with truncate_time flag."""
        decoded = util.decode_datetime('2012-09-30T15:31:50.262343123', truncate_time=True)
        expected = datetime.datetime(2012, 9, 30, 15, 31, 50, 262343)
        self.assertEquals(expected, decoded)

    def testDateTimeTimeZones(self):
        """Test that a datetime string with a timezone is decoded correctly."""
        tests = (('2012-09-30T15:31:50.262-06:00', (2012, 9, 30, 15, 31, 50, 262000, util.TimeZoneOffset(-360))), ('2012-09-30T15:31:50.262+01:30', (2012, 9, 30, 15, 31, 50, 262000, util.TimeZoneOffset(90))), ('2012-09-30T15:31:50+00:05', (2012, 9, 30, 15, 31, 50, 0, util.TimeZoneOffset(5))), ('2012-09-30T15:31:50+00:00', (2012, 9, 30, 15, 31, 50, 0, util.TimeZoneOffset(0))), ('2012-09-30t15:31:50-00:00', (2012, 9, 30, 15, 31, 50, 0, util.TimeZoneOffset(0))), ('2012-09-30t15:31:50z', (2012, 9, 30, 15, 31, 50, 0, util.TimeZoneOffset(0))), ('2012-09-30T15:31:50-23:00', (2012, 9, 30, 15, 31, 50, 0, util.TimeZoneOffset(-1380))))
        for datetime_string, datetime_vals in tests:
            decoded = util.decode_datetime(datetime_string)
            expected = datetime.datetime(*datetime_vals)
            self.assertEquals(expected, decoded)

    def testDecodeDateTimeInvalid(self):
        """Test that decoding malformed datetime strings raises execptions."""
        for datetime_string in ('invalid', '2012-09-30T15:31:50.', '-08:00 2012-09-30T15:31:50.262', '2012-09-30T15:31', '2012-09-30T15:31Z', '2012-09-30T15:31:50ZZ', '2012-09-30T15:31:50.262 blah blah -08:00', '1000-99-99T25:99:99.999-99:99', '2012-09-30T15:31:50.262343123'):
            self.assertRaises(ValueError, util.decode_datetime, datetime_string)

    def testTimeZoneOffsetDelta(self):
        """Test that delta works with TimeZoneOffset."""
        time_zone = util.TimeZoneOffset(datetime.timedelta(minutes=3))
        epoch = time_zone.utcoffset(datetime.datetime.utcfromtimestamp(0))
        self.assertEqual(180, util.total_seconds(epoch))