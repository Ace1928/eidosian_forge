import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
class ParseStringTests(TestCase):

    def test_quoted(self):
        self.assertEqual(b' foo', _parse_string(b'" foo"'))
        self.assertEqual(b'\tfoo', _parse_string(b'"\\tfoo"'))

    def test_not_quoted(self):
        self.assertEqual(b'foo', _parse_string(b'foo'))
        self.assertEqual(b'foo bar', _parse_string(b'foo bar'))

    def test_nothing(self):
        self.assertEqual(b'', _parse_string(b''))

    def test_tab(self):
        self.assertEqual(b'\tbar\t', _parse_string(b'\\tbar\\t'))

    def test_newline(self):
        self.assertEqual(b'\nbar\t', _parse_string(b'\\nbar\\t\t'))

    def test_quote(self):
        self.assertEqual(b'"foo"', _parse_string(b'\\"foo\\"'))