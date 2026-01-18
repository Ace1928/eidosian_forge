import operator
import sys
import types
import unittest
import abc
import pytest
import six
class EnsureTests:
    UNICODE_EMOJI = six.u('😀')
    BINARY_EMOJI = b'\xf0\x9f\x98\x80'

    def test_ensure_binary_raise_type_error(self):
        with pytest.raises(TypeError):
            six.ensure_str(8)

    def test_errors_and_encoding(self):
        six.ensure_binary(self.UNICODE_EMOJI, encoding='latin-1', errors='ignore')
        with pytest.raises(UnicodeEncodeError):
            six.ensure_binary(self.UNICODE_EMOJI, encoding='latin-1', errors='strict')

    def test_ensure_binary_raise(self):
        converted_unicode = six.ensure_binary(self.UNICODE_EMOJI, encoding='utf-8', errors='strict')
        converted_binary = six.ensure_binary(self.BINARY_EMOJI, encoding='utf-8', errors='strict')
        if six.PY2:
            assert converted_unicode == self.BINARY_EMOJI and isinstance(converted_unicode, str)
            assert converted_binary == self.BINARY_EMOJI and isinstance(converted_binary, str)
        else:
            assert converted_unicode == self.BINARY_EMOJI and isinstance(converted_unicode, bytes)
            assert converted_binary == self.BINARY_EMOJI and isinstance(converted_binary, bytes)

    def test_ensure_str(self):
        converted_unicode = six.ensure_str(self.UNICODE_EMOJI, encoding='utf-8', errors='strict')
        converted_binary = six.ensure_str(self.BINARY_EMOJI, encoding='utf-8', errors='strict')
        if six.PY2:
            assert converted_unicode == self.BINARY_EMOJI and isinstance(converted_unicode, str)
            assert converted_binary == self.BINARY_EMOJI and isinstance(converted_binary, str)
        else:
            assert converted_unicode == self.UNICODE_EMOJI and isinstance(converted_unicode, str)
            assert converted_binary == self.UNICODE_EMOJI and isinstance(converted_unicode, str)

    def test_ensure_text(self):
        converted_unicode = six.ensure_text(self.UNICODE_EMOJI, encoding='utf-8', errors='strict')
        converted_binary = six.ensure_text(self.BINARY_EMOJI, encoding='utf-8', errors='strict')
        if six.PY2:
            assert converted_unicode == self.UNICODE_EMOJI and isinstance(converted_unicode, unicode)
            assert converted_binary == self.UNICODE_EMOJI and isinstance(converted_unicode, unicode)
        else:
            assert converted_unicode == self.UNICODE_EMOJI and isinstance(converted_unicode, str)
            assert converted_binary == self.UNICODE_EMOJI and isinstance(converted_unicode, str)