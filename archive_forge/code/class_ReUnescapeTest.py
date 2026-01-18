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
class ReUnescapeTest(unittest.TestCase):

    def test_re_unescape(self):
        test_strings = ('/favicon.ico', 'index.html', 'Hello, World!', '!$@#%;')
        for string in test_strings:
            self.assertEqual(string, re_unescape(re.escape(string)))

    def test_re_unescape_raises_error_on_invalid_input(self):
        with self.assertRaises(ValueError):
            re_unescape('\\d')
        with self.assertRaises(ValueError):
            re_unescape('\\b')
        with self.assertRaises(ValueError):
            re_unescape('\\Z')