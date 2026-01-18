from io import StringIO
import re
import sys
import datetime
import unittest
import tornado
from tornado.escape import utf8
from tornado.util import (
import typing
from typing import cast
class TimedeltaToSecondsTest(unittest.TestCase):

    def test_timedelta_to_seconds(self):
        time_delta = datetime.timedelta(hours=1)
        self.assertEqual(timedelta_to_seconds(time_delta), 3600.0)