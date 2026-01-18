import unittest
from io import BytesIO
from testtools.compat import _b
import subunit.chunked
class TestDecode(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.output = BytesIO()
        self.decoder = subunit.chunked.Decoder(self.output)

    def test_close_read_length_short_errors(self):
        self.assertRaises(ValueError, self.decoder.close)

    def test_close_body_short_errors(self):
        self.assertEqual(None, self.decoder.write(_b('2\r\na')))
        self.assertRaises(ValueError, self.decoder.close)

    def test_close_body_buffered_data_errors(self):
        self.assertEqual(None, self.decoder.write(_b('2\r')))
        self.assertRaises(ValueError, self.decoder.close)

    def test_close_after_finished_stream_safe(self):
        self.assertEqual(None, self.decoder.write(_b('2\r\nab')))
        self.assertEqual(_b(''), self.decoder.write(_b('0\r\n')))
        self.decoder.close()

    def test_decode_nothing(self):
        self.assertEqual(_b(''), self.decoder.write(_b('0\r\n')))
        self.assertEqual(_b(''), self.output.getvalue())

    def test_decode_serialised_form(self):
        self.assertEqual(None, self.decoder.write(_b('F\r\n')))
        self.assertEqual(None, self.decoder.write(_b('serialised\n')))
        self.assertEqual(_b(''), self.decoder.write(_b('form0\r\n')))

    def test_decode_short(self):
        self.assertEqual(_b(''), self.decoder.write(_b('3\r\nabc0\r\n')))
        self.assertEqual(_b('abc'), self.output.getvalue())

    def test_decode_combines_short(self):
        self.assertEqual(_b(''), self.decoder.write(_b('6\r\nabcdef0\r\n')))
        self.assertEqual(_b('abcdef'), self.output.getvalue())

    def test_decode_excess_bytes_from_write(self):
        self.assertEqual(_b('1234'), self.decoder.write(_b('3\r\nabc0\r\n1234')))
        self.assertEqual(_b('abc'), self.output.getvalue())

    def test_decode_write_after_finished_errors(self):
        self.assertEqual(_b('1234'), self.decoder.write(_b('3\r\nabc0\r\n1234')))
        self.assertRaises(ValueError, self.decoder.write, _b(''))

    def test_decode_hex(self):
        self.assertEqual(_b(''), self.decoder.write(_b('A\r\n12345678900\r\n')))
        self.assertEqual(_b('1234567890'), self.output.getvalue())

    def test_decode_long_ranges(self):
        self.assertEqual(None, self.decoder.write(_b('10000\r\n')))
        self.assertEqual(None, self.decoder.write(_b('1' * 65536)))
        self.assertEqual(None, self.decoder.write(_b('10000\r\n')))
        self.assertEqual(None, self.decoder.write(_b('2' * 65536)))
        self.assertEqual(_b(''), self.decoder.write(_b('0\r\n')))
        self.assertEqual(_b('1' * 65536 + '2' * 65536), self.output.getvalue())

    def test_decode_newline_nonstrict(self):
        """Tolerate chunk markers with no CR character."""
        self.decoder = subunit.chunked.Decoder(self.output, strict=False)
        self.assertEqual(None, self.decoder.write(_b('a\n')))
        self.assertEqual(None, self.decoder.write(_b('abcdeabcde')))
        self.assertEqual(_b(''), self.decoder.write(_b('0\n')))
        self.assertEqual(_b('abcdeabcde'), self.output.getvalue())

    def test_decode_strict_newline_only(self):
        """Reject chunk markers with no CR character in strict mode."""
        self.assertRaises(ValueError, self.decoder.write, _b('a\n'))

    def test_decode_strict_multiple_crs(self):
        self.assertRaises(ValueError, self.decoder.write, _b('a\r\r\n'))

    def test_decode_short_header(self):
        self.assertRaises(ValueError, self.decoder.write, _b('\n'))