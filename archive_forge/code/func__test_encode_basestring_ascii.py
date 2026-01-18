from unittest import TestCase
import simplejson.encoder
from simplejson.compat import b
def _test_encode_basestring_ascii(self, encode_basestring_ascii):
    fname = encode_basestring_ascii.__name__
    for input_string, expect in CASES:
        result = encode_basestring_ascii(input_string)
        self.assertEqual(result, expect, '%r != %r for %s(%r)' % (result, expect, fname, input_string))