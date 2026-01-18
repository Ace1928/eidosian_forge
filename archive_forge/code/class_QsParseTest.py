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
class QsParseTest(unittest.TestCase):

    def test_parsing(self):
        qsstring = 'a=1&b=2&a=3'
        qs = urllib.parse.parse_qs(qsstring)
        qsl = list(qs_to_qsl(qs))
        self.assertIn(('a', '1'), qsl)
        self.assertIn(('a', '3'), qsl)
        self.assertIn(('b', '2'), qsl)