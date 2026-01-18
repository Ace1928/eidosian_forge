import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
class EscapeValueTests(TestCase):

    def test_nothing(self):
        self.assertEqual(b'foo', _escape_value(b'foo'))

    def test_backslash(self):
        self.assertEqual(b'foo\\\\', _escape_value(b'foo\\'))

    def test_newline(self):
        self.assertEqual(b'foo\\n', _escape_value(b'foo\n'))