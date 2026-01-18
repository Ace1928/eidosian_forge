import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
class DerSetOfTests(unittest.TestCase):

    def testInit1(self):
        der = DerSetOf([DerInteger(1), DerInteger(2)])
        self.assertEqual(der.encode(), b('1\x06\x02\x01\x01\x02\x01\x02'))

    def testEncode1(self):
        der = DerSetOf()
        self.assertEqual(der.encode(), b('1\x00'))
        der.add(0)
        self.assertEqual(der.encode(), b('1\x03\x02\x01\x00'))
        self.assertEqual(der.encode(), b('1\x03\x02\x01\x00'))

    def testEncode2(self):
        der = DerSetOf()
        der.add(384)
        der.add(255)
        self.assertEqual(der.encode(), b('1\x08\x02\x02\x00ÿ\x02\x02\x01\x80'))
        der = DerSetOf([384, 255])
        self.assertEqual(der.encode(), b('1\x08\x02\x02\x00ÿ\x02\x02\x01\x80'))

    def testEncode3(self):
        der = DerSetOf()
        der.add(384)
        self.assertRaises(ValueError, der.add, b('\x00\x02\x00\x00'))

    def testEncode4(self):
        der = DerSetOf()
        der.add(b('\x01\x00'))
        der.add(b('\x01\x01\x01'))
        self.assertEqual(der.encode(), b('1\x05\x01\x00\x01\x01\x01'))

    def testDecode1(self):
        der = DerSetOf()
        der.decode(b('1\x00'))
        self.assertEqual(len(der), 0)
        der.decode(b('1\x03\x02\x01\x00'))
        self.assertEqual(len(der), 1)
        self.assertEqual(list(der), [0])

    def testDecode2(self):
        der = DerSetOf()
        der.decode(b('1\x08\x02\x02\x01\x80\x02\x02\x00ÿ'))
        self.assertEqual(len(der), 2)
        l = list(der)
        self.assertTrue(384 in l)
        self.assertTrue(255 in l)

    def testDecode3(self):
        der = DerSetOf()
        self.assertRaises(ValueError, der.decode, b('0\n\x02\x02\x01\x80$\x02¶c\x12\x00'))

    def testDecode4(self):
        der = DerSetOf()
        self.assertEqual(der, der.decode(b('1\x08\x02\x02\x01\x80\x02\x02\x00ÿ')))

    def testErrDecode1(self):
        der = DerSetOf()
        self.assertRaises(ValueError, der.decode, b('1\x08\x02\x02\x01\x80\x02\x02\x00ÿª'))