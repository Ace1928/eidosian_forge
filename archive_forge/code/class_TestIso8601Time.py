import datetime
import logging
import time
from unittest import mock
import iso8601
from oslotest import base as test_base
from testtools import matchers
from oslo_utils import timeutils
class TestIso8601Time(test_base.BaseTestCase):

    def _instaneous(self, timestamp, yr, mon, day, hr, minute, sec, micro):
        self.assertEqual(timestamp.year, yr)
        self.assertEqual(timestamp.month, mon)
        self.assertEqual(timestamp.day, day)
        self.assertEqual(timestamp.hour, hr)
        self.assertEqual(timestamp.minute, minute)
        self.assertEqual(timestamp.second, sec)
        self.assertEqual(timestamp.microsecond, micro)

    def _do_test(self, time_str, yr, mon, day, hr, minute, sec, micro, shift):
        DAY_SECONDS = 24 * 60 * 60
        timestamp = timeutils.parse_isotime(time_str)
        self._instaneous(timestamp, yr, mon, day, hr, minute, sec, micro)
        offset = timestamp.tzinfo.utcoffset(None)
        self.assertEqual(offset.seconds + offset.days * DAY_SECONDS, shift)

    def test_zulu(self):
        time_str = '2012-02-14T20:53:07Z'
        self._do_test(time_str, 2012, 2, 14, 20, 53, 7, 0, 0)

    def test_zulu_micros(self):
        time_str = '2012-02-14T20:53:07.123Z'
        self._do_test(time_str, 2012, 2, 14, 20, 53, 7, 123000, 0)

    def test_offset_east(self):
        time_str = '2012-02-14T20:53:07+04:30'
        offset = 4.5 * 60 * 60
        self._do_test(time_str, 2012, 2, 14, 20, 53, 7, 0, offset)

    def test_offset_east_micros(self):
        time_str = '2012-02-14T20:53:07.42+04:30'
        offset = 4.5 * 60 * 60
        self._do_test(time_str, 2012, 2, 14, 20, 53, 7, 420000, offset)

    def test_offset_west(self):
        time_str = '2012-02-14T20:53:07-05:30'
        offset = -5.5 * 60 * 60
        self._do_test(time_str, 2012, 2, 14, 20, 53, 7, 0, offset)

    def test_offset_west_micros(self):
        time_str = '2012-02-14T20:53:07.654321-05:30'
        offset = -5.5 * 60 * 60
        self._do_test(time_str, 2012, 2, 14, 20, 53, 7, 654321, offset)

    def test_compare(self):
        zulu = timeutils.parse_isotime('2012-02-14T20:53:07')
        east = timeutils.parse_isotime('2012-02-14T20:53:07-01:00')
        west = timeutils.parse_isotime('2012-02-14T20:53:07+01:00')
        self.assertTrue(east > west)
        self.assertTrue(east > zulu)
        self.assertTrue(zulu > west)

    def test_compare_micros(self):
        zulu = timeutils.parse_isotime('2012-02-14T20:53:07.6544')
        east = timeutils.parse_isotime('2012-02-14T19:53:07.654321-01:00')
        west = timeutils.parse_isotime('2012-02-14T21:53:07.655+01:00')
        self.assertTrue(east < west)
        self.assertTrue(east < zulu)
        self.assertTrue(zulu < west)

    def test_zulu_normalize(self):
        time_str = '2012-02-14T20:53:07Z'
        zulu = timeutils.parse_isotime(time_str)
        normed = timeutils.normalize_time(zulu)
        self._instaneous(normed, 2012, 2, 14, 20, 53, 7, 0)

    def test_east_normalize(self):
        time_str = '2012-02-14T20:53:07-07:00'
        east = timeutils.parse_isotime(time_str)
        normed = timeutils.normalize_time(east)
        self._instaneous(normed, 2012, 2, 15, 3, 53, 7, 0)

    def test_west_normalize(self):
        time_str = '2012-02-14T20:53:07+21:00'
        west = timeutils.parse_isotime(time_str)
        normed = timeutils.normalize_time(west)
        self._instaneous(normed, 2012, 2, 13, 23, 53, 7, 0)

    def test_normalize_aware_to_naive(self):
        dt = datetime.datetime(2011, 2, 14, 20, 53, 7)
        time_str = '2011-02-14T20:53:07+21:00'
        aware = timeutils.parse_isotime(time_str)
        naive = timeutils.normalize_time(aware)
        self.assertTrue(naive < dt)

    def test_normalize_zulu_aware_to_naive(self):
        dt = datetime.datetime(2011, 2, 14, 20, 53, 7)
        time_str = '2011-02-14T19:53:07Z'
        aware = timeutils.parse_isotime(time_str)
        naive = timeutils.normalize_time(aware)
        self.assertTrue(naive < dt)

    def test_normalize_naive(self):
        dt = datetime.datetime(2011, 2, 14, 20, 53, 7)
        dtn = datetime.datetime(2011, 2, 14, 19, 53, 7)
        naive = timeutils.normalize_time(dtn)
        self.assertTrue(naive < dt)