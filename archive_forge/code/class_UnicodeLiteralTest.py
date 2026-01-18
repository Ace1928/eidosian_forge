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
class UnicodeLiteralTest(unittest.TestCase):

    def test_unicode_escapes(self):
        self.assertEqual(utf8('é'), b'\xc3\xa9')