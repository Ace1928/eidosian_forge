import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
class FormatStringTests(TestCase):

    def test_quoted(self):
        self.assertEqual(b'" foo"', _format_string(b' foo'))
        self.assertEqual(b'"\\tfoo"', _format_string(b'\tfoo'))

    def test_not_quoted(self):
        self.assertEqual(b'foo', _format_string(b'foo'))
        self.assertEqual(b'foo bar', _format_string(b'foo bar'))