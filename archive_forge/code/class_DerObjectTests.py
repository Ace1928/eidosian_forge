import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
class DerObjectTests(unittest.TestCase):

    def testObjInit1(self):
        self.assertRaises(ValueError, DerObject, b('\x00\x99'))
        self.assertRaises(ValueError, DerObject, 31)

    def testObjEncode1(self):
        der = DerObject(b('\x02'))
        self.assertEqual(der.encode(), b('\x02\x00'))
        der.payload = b('E')
        self.assertEqual(der.encode(), b('\x02\x01E'))
        self.assertEqual(der.encode(), b('\x02\x01E'))
        der = DerObject(4)
        der.payload = b('E')
        self.assertEqual(der.encode(), b('\x04\x01E'))
        der = DerObject(b('\x10'), constructed=True)
        self.assertEqual(der.encode(), b('0\x00'))

    def testObjEncode2(self):
        der = DerObject(3, b('\x12\x12'))
        self.assertEqual(der.encode(), b('\x03\x02\x12\x12'))

    def testObjEncode3(self):
        der = DerObject(b('\x10'))
        der.payload = b('0') * 128
        self.assertEqual(der.encode(), b('\x10\x81\x80' + '0' * 128))

    def testObjEncode4(self):
        der = DerObject(16, implicit=1, constructed=True)
        der.payload = b('ppll')
        self.assertEqual(der.encode(), b('¡\x04ppll'))
        der = DerObject(2, implicit=30, constructed=False)
        der.payload = b('ppll')
        self.assertEqual(der.encode(), b('\x9e\x04ppll'))

    def testObjEncode5(self):
        der = DerObject(16, explicit=5)
        der.payload = b('xxll')
        self.assertEqual(der.encode(), b('¥\x06\x10\x04xxll'))

    def testObjDecode1(self):
        der = DerObject(2)
        der.decode(b('\x02\x02\x01\x02'))
        self.assertEqual(der.payload, b('\x01\x02'))
        self.assertEqual(der._tag_octet, 2)

    def testObjDecode2(self):
        der = DerObject(2)
        der.decode(b('\x02\x81\x80' + '1' * 128))
        self.assertEqual(der.payload, b('1') * 128)
        self.assertEqual(der._tag_octet, 2)

    def testObjDecode3(self):
        der = DerObject(2)
        self.assertRaises(ValueError, der.decode, b('\x02\x02\x01\x02ÿ'))
        der = DerObject(2)
        self.assertRaises(ValueError, der.decode, b('\x02\x02\x01'))

    def testObjDecode4(self):
        der = DerObject(2, constructed=False, implicit=15)
        self.assertRaises(ValueError, der.decode, b('\x02\x02\x01\x02'))
        der.decode(b('\x8f\x01\x00'))
        self.assertEqual(der.payload, b('\x00'))
        der = DerObject(2, constructed=True, implicit=15)
        self.assertRaises(ValueError, der.decode, b('\x02\x02\x01\x02'))
        der.decode(b('¯\x01\x00'))
        self.assertEqual(der.payload, b('\x00'))

    def testObjDecode5(self):
        der = DerObject(2)
        self.assertRaises(ValueError, der.decode, b('\x03\x02\x01\x02'))

    def testObjDecode6(self):
        der = DerObject()
        der.decode(b('e\x01\x88'))
        self.assertEqual(der._tag_octet, 101)
        self.assertEqual(der.payload, b('\x88'))

    def testObjDecode7(self):
        der = DerObject(16, explicit=5)
        der.decode(b('¥\x06\x10\x04xxll'))
        self.assertEqual(der._inner_tag_octet, 16)
        self.assertEqual(der.payload, b('xxll'))
        der = DerObject(16, explicit=0)
        der.decode(b('\xa0\x06\x10\x04xxll'))
        self.assertEqual(der._inner_tag_octet, 16)
        self.assertEqual(der.payload, b('xxll'))

    def testObjDecode8(self):
        der = DerObject(2)
        self.assertEqual(der, der.decode(b('\x02\x02\x01\x02')))