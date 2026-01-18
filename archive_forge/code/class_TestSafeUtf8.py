import errno
import os
import select
import socket
import sys
import tempfile
import time
from io import BytesIO
from .. import errors, osutils, tests, trace, win32utils
from . import features, file_utils, test__walkdirs_win32
from .scenarios import load_tests_apply_scenarios
class TestSafeUtf8(tests.TestCase):

    def test_from_ascii_string(self):
        f = b'foobar'
        self.assertEqual(b'foobar', osutils.safe_utf8(f))

    def test_from_unicode_string_ascii_contents(self):
        self.assertEqual(b'bargam', osutils.safe_utf8('bargam'))

    def test_from_unicode_string_unicode_contents(self):
        self.assertEqual(b'bargam\xc2\xae', osutils.safe_utf8('bargam®'))

    def test_from_utf8_string(self):
        self.assertEqual(b'foo\xc2\xae', osutils.safe_utf8(b'foo\xc2\xae'))

    def test_bad_utf8_string(self):
        self.assertRaises(errors.BzrBadParameterNotUnicode, osutils.safe_utf8, b'\xbb\xbb')