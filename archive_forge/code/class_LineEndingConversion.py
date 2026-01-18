from dulwich.tests import TestCase
from ..line_ending import (
from ..objects import Blob
class LineEndingConversion(TestCase):
    """Test the line ending conversion functions in various cases."""

    def test_convert_crlf_to_lf_no_op(self):
        self.assertEqual(convert_crlf_to_lf(b'foobar'), b'foobar')

    def test_convert_crlf_to_lf(self):
        self.assertEqual(convert_crlf_to_lf(b'line1\r\nline2'), b'line1\nline2')

    def test_convert_crlf_to_lf_mixed(self):
        self.assertEqual(convert_crlf_to_lf(b'line1\r\n\nline2'), b'line1\n\nline2')

    def test_convert_lf_to_crlf_no_op(self):
        self.assertEqual(convert_lf_to_crlf(b'foobar'), b'foobar')

    def test_convert_lf_to_crlf(self):
        self.assertEqual(convert_lf_to_crlf(b'line1\nline2'), b'line1\r\nline2')

    def test_convert_lf_to_crlf_mixed(self):
        self.assertEqual(convert_lf_to_crlf(b'line1\r\n\nline2'), b'line1\r\n\r\nline2')