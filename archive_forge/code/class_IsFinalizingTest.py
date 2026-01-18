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
class IsFinalizingTest(unittest.TestCase):

    def test_basic(self):
        self.assertFalse(is_finalizing())