from unittest import TestCase
import simplejson.encoder
from simplejson.compat import b
class TestEncodeBaseStringAscii(TestCase):

    def test_py_encode_basestring_ascii(self):
        self._test_encode_basestring_ascii(simplejson.encoder.py_encode_basestring_ascii)

    def test_c_encode_basestring_ascii(self):
        if not simplejson.encoder.c_encode_basestring_ascii:
            return
        self._test_encode_basestring_ascii(simplejson.encoder.c_encode_basestring_ascii)

    def _test_encode_basestring_ascii(self, encode_basestring_ascii):
        fname = encode_basestring_ascii.__name__
        for input_string, expect in CASES:
            result = encode_basestring_ascii(input_string)
            self.assertEqual(result, expect, '%r != %r for %s(%r)' % (result, expect, fname, input_string))

    def test_sorted_dict(self):
        items = [('one', 1), ('two', 2), ('three', 3), ('four', 4), ('five', 5)]
        s = simplejson.dumps(dict(items), sort_keys=True)
        self.assertEqual(s, '{"five": 5, "four": 4, "one": 1, "three": 3, "two": 2}')