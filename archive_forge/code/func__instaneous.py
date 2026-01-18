import datetime
import logging
import time
from unittest import mock
import iso8601
from oslotest import base as test_base
from testtools import matchers
from oslo_utils import timeutils
def _instaneous(self, timestamp, yr, mon, day, hr, minute, sec, micro):
    self.assertEqual(timestamp.year, yr)
    self.assertEqual(timestamp.month, mon)
    self.assertEqual(timestamp.day, day)
    self.assertEqual(timestamp.hour, hr)
    self.assertEqual(timestamp.minute, minute)
    self.assertEqual(timestamp.second, sec)
    self.assertEqual(timestamp.microsecond, micro)