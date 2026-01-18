from datetime import (
import time
import unittest
class LocalTimeTestCase(unittest.TestCase):
    """
    Test the use of the timezone saved locally. Since it is hard to test using
    doctest.
    """

    def setUp(self):
        local_utcoffset = _utc_offset(datetime.now(), use_system_timezone=True)
        self.local_utcoffset = timedelta(seconds=local_utcoffset)
        self.local_timezone = _timezone(local_utcoffset)

    def test_datetime(self):
        d = datetime.now()
        self.assertEqual(rfc3339(d), d.strftime('%Y-%m-%dT%H:%M:%S') + self.local_timezone)

    def test_datetime_timezone(self):

        class FixedNoDst(tzinfo):
            """A timezone info with fixed offset, not DST"""

            def utcoffset(self, dt):
                return timedelta(hours=2, minutes=30)

            def dst(self, dt):
                return None
        fixed_no_dst = FixedNoDst()

        class Fixed(FixedNoDst):
            """A timezone info with DST"""

            def utcoffset(self, dt):
                return timedelta(hours=3, minutes=15)

            def dst(self, dt):
                return timedelta(hours=3, minutes=15)
        fixed = Fixed()
        d = datetime.now().replace(tzinfo=fixed_no_dst)
        timezone = _timezone(_timedelta_to_seconds(fixed_no_dst.utcoffset(None)))
        self.assertEqual(rfc3339(d), d.strftime('%Y-%m-%dT%H:%M:%S') + timezone)
        d = datetime.now().replace(tzinfo=fixed)
        timezone = _timezone(_timedelta_to_seconds(fixed.dst(None)))
        self.assertEqual(rfc3339(d), d.strftime('%Y-%m-%dT%H:%M:%S') + timezone)

    def test_datetime_utc(self):
        d = datetime.now()
        d_utc = d - self.local_utcoffset
        self.assertEqual(rfc3339(d, utc=True), d_utc.strftime('%Y-%m-%dT%H:%M:%SZ'))

    def test_date(self):
        d = date.today()
        self.assertEqual(rfc3339(d), d.strftime('%Y-%m-%dT%H:%M:%S') + self.local_timezone)

    def test_date_utc(self):
        d = date.today()
        d_utc = datetime(*d.timetuple()[:3]) - self.local_utcoffset
        self.assertEqual(rfc3339(d, utc=True), d_utc.strftime('%Y-%m-%dT%H:%M:%SZ'))

    def test_timestamp(self):
        d = time.time()
        self.assertEqual(rfc3339(d), datetime.fromtimestamp(d).strftime('%Y-%m-%dT%H:%M:%S') + self.local_timezone)

    def test_timestamp_utc(self):
        d = time.time()
        d_utc = datetime.utcfromtimestamp(d) + self.local_utcoffset
        self.assertEqual(rfc3339(d), d_utc.strftime('%Y-%m-%dT%H:%M:%S') + self.local_timezone)

    def test_before_1970(self):
        d = date(1885, 1, 4)
        self.assertTrue(rfc3339(d).startswith('1885-01-04T00:00:00'))
        self.assertEqual(rfc3339(d, utc=True, use_system_timezone=False), '1885-01-04T00:00:00Z')

    def test_1920(self):
        d = date(1920, 2, 29)
        x = rfc3339(d, utc=False, use_system_timezone=True)
        self.assertTrue(x.startswith('1920-02-29T00:00:00'))
    if 'PST' in time.tzname:

        def testPDTChange(self):
            """Test Daylight saving change"""
            self.assertEqual(rfc3339(datetime(2010, 3, 14, 1, 59)), '2010-03-14T01:59:00-08:00')
            self.assertEqual(rfc3339(datetime(2010, 3, 14, 3, 0)), '2010-03-14T03:00:00-07:00')

        def testPSTChange(self):
            """Test Standard time change"""
            self.assertEqual(rfc3339(datetime(2010, 11, 7, 0, 59)), '2010-11-07T00:59:00-07:00')
            self.assertEqual(rfc3339(datetime(2010, 11, 7, 1, 0)), '2010-11-07T01:00:00-07:00')

    def test_millisecond(self):
        x = datetime(2018, 9, 20, 13, 11, 21, 123000)
        self.assertEqual(format_millisecond(datetime(2018, 9, 20, 13, 11, 21, 123000), utc=True, use_system_timezone=False), '2018-09-20T13:11:21.123Z')

    def test_microsecond(self):
        x = datetime(2018, 9, 20, 13, 11, 21, 12345)
        self.assertEqual(format_microsecond(datetime(2018, 9, 20, 13, 11, 21, 12345), utc=True, use_system_timezone=False), '2018-09-20T13:11:21.012345Z')