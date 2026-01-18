from tornado.httputil import (
from tornado.escape import utf8, native_str
from tornado.log import gen_log
from tornado.testing import ExpectLog
from tornado.test.util import ignore_deprecation
import copy
import datetime
import logging
import pickle
import time
import urllib.parse
import unittest
from typing import Tuple, Dict, List
class FormatTimestampTest(unittest.TestCase):
    TIMESTAMP = 1359312200.503611
    EXPECTED = 'Sun, 27 Jan 2013 18:43:20 GMT'

    def check(self, value):
        self.assertEqual(format_timestamp(value), self.EXPECTED)

    def test_unix_time_float(self):
        self.check(self.TIMESTAMP)

    def test_unix_time_int(self):
        self.check(int(self.TIMESTAMP))

    def test_struct_time(self):
        self.check(time.gmtime(self.TIMESTAMP))

    def test_time_tuple(self):
        tup = tuple(time.gmtime(self.TIMESTAMP))
        self.assertEqual(9, len(tup))
        self.check(tup)

    def test_utc_naive_datetime(self):
        self.check(datetime.datetime.fromtimestamp(self.TIMESTAMP, datetime.timezone.utc).replace(tzinfo=None))

    def test_utc_naive_datetime_deprecated(self):
        with ignore_deprecation():
            self.check(datetime.datetime.utcfromtimestamp(self.TIMESTAMP))

    def test_utc_aware_datetime(self):
        self.check(datetime.datetime.fromtimestamp(self.TIMESTAMP, datetime.timezone.utc))

    def test_other_aware_datetime(self):
        self.check(datetime.datetime.fromtimestamp(self.TIMESTAMP, datetime.timezone(datetime.timedelta(hours=-4))))