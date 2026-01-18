import pathlib
import unittest
from pygame import encode_string, encode_file_path
class RWopsEncodeFilePathTest(unittest.TestCase):

    def test_encoding(self):
        u = 'Hello'
        encoded_file_path = encode_file_path(u)
        self.assertIsInstance(encoded_file_path, bytes)

    def test_error_fowarding(self):
        self.assertRaises(SyntaxError, encode_file_path)

    def test_path_with_null_bytes(self):
        b = b'a\x00b\x00c'
        encoded_file_path = encode_file_path(b)
        self.assertIsNone(encoded_file_path)

    def test_etype(self):
        b = b'a\x00b\x00c'
        self.assertRaises(TypeError, encode_file_path, b, TypeError)

    def test_etype__invalid(self):
        """Ensures invalid etypes are properly handled."""
        for etype in ('SyntaxError', self):
            self.assertRaises(TypeError, encode_file_path, 'test', etype)