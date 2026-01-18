import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
class DerBitStringTests(unittest.TestCase):

    def testInit1(self):
        der = DerBitString(b('ÿ'))
        self.assertEqual(der.encode(), b('\x03\x02\x00ÿ'))

    def testInit2(self):
        der = DerBitString(DerInteger(1))
        self.assertEqual(der.encode(), b('\x03\x04\x00\x02\x01\x01'))

    def testEncode1(self):
        der = DerBitString()
        self.assertEqual(der.encode(), b('\x03\x01\x00'))
        der = DerBitString(b('\x01\x02'))
        self.assertEqual(der.encode(), b('\x03\x03\x00\x01\x02'))
        der = DerBitString()
        der.value = b('\x01\x02')
        self.assertEqual(der.encode(), b('\x03\x03\x00\x01\x02'))

    def testDecode1(self):
        der = DerBitString()
        der.decode(b('\x03\x00'))
        self.assertEqual(der.value, b(''))
        der.decode(b('\x03\x03\x00\x01\x02'))
        self.assertEqual(der.value, b('\x01\x02'))

    def testDecode2(self):
        der = DerBitString()
        self.assertEqual(der, der.decode(b('\x03\x00')))