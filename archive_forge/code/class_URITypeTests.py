import re
import unittest
from oslo_config import types
class URITypeTests(TypeTestHelper, unittest.TestCase):
    type = types.URI()

    def test_uri(self):
        self.assertConvertedValue('http://example.com', 'http://example.com')
        self.assertInvalid('invalid')
        self.assertInvalid('http://')

    def test_repr(self):
        self.assertEqual('URI', repr(types.URI()))

    def test_max_length(self):
        self.type_instance = types.String(max_length=30)
        self.assertInvalid('http://www.example.com/versions')
        self.assertConvertedValue('http://www.example.com', 'http://www.example.com')

    def test_equality(self):
        a = types.URI()
        b = types.URI()
        self.assertEqual(a, b)

    def test_equality_length(self):
        a = types.URI(max_length=5)
        b = types.URI(max_length=5)
        self.assertEqual(a, b)

    def test_equality_length_not(self):
        a = types.URI()
        b = types.URI(max_length=5)
        c = types.URI(max_length=10)
        self.assertNotEqual(a, b)
        self.assertNotEqual(c, b)

    def test_equality_schemes(self):
        a = types.URI(schemes=['ftp'])
        b = types.URI(schemes=['ftp'])
        self.assertEqual(a, b)

    def test_equality_schemes_not(self):
        a = types.URI()
        b = types.URI(schemes=['ftp'])
        c = types.URI(schemes=['http'])
        self.assertNotEqual(a, b)
        self.assertNotEqual(c, b)