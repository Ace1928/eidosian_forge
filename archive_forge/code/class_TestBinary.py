from decimal import Decimal
from tests.compat import unittest
from boto.compat import six, json
from boto.dynamodb import types
from boto.dynamodb.exceptions import DynamoDBNumberError
class TestBinary(unittest.TestCase):

    def test_good_input(self):
        data = types.Binary(b'\x01')
        self.assertEqual(b'\x01', data)
        self.assertEqual(b'\x01', bytes(data))

    def test_non_ascii_good_input(self):
        data = types.Binary(b'\x88')
        self.assertEqual(b'\x88', data)
        self.assertEqual(b'\x88', bytes(data))

    @unittest.skipUnless(six.PY2, 'Python 2 only')
    def test_bad_input(self):
        with self.assertRaises(TypeError):
            types.Binary(1)

    @unittest.skipUnless(six.PY3, 'Python 3 only')
    def test_bytes_input(self):
        data = types.Binary(1)
        self.assertEqual(data, b'\x00')
        self.assertEqual(data.value, b'\x00')

    @unittest.skipUnless(six.PY2, 'Python 2 only')
    def test_unicode_py2(self):
        data = types.Binary(u'\x01')
        self.assertEqual(data, b'\x01')
        self.assertEqual(bytes(data), b'\x01')
        self.assertEqual(data, u'\x01')
        self.assertEqual(type(data.value), bytes)

    @unittest.skipUnless(six.PY3, 'Python 3 only')
    def test_unicode_py3(self):
        with self.assertRaises(TypeError):
            types.Binary(u'\x01')