from unittest import main, TestCase, TestSuite
from binascii import unhexlify, hexlify
from Cryptodome.Util.py3compat import *
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Protocol.SecretSharing import Shamir, _Element, \
class Element_Tests(TestCase):

    def test1(self):
        e = _Element(256)
        self.assertEqual(int(e), 256)
        self.assertEqual(e.encode(), bchr(0) * 14 + b('\x01\x00'))
        e = _Element(bchr(0) * 14 + b('\x01\x10'))
        self.assertEqual(int(e), 272)
        self.assertEqual(e.encode(), bchr(0) * 14 + b('\x01\x10'))
        self.assertRaises(ValueError, _Element, bchr(0))

    def test2(self):
        e = _Element(16)
        f = _Element(10)
        self.assertEqual(int(e + f), 26)

    def test3(self):
        zero = _Element(0)
        one = _Element(1)
        two = _Element(2)
        x = _Element(6) * zero
        self.assertEqual(int(x), 0)
        x = _Element(6) * one
        self.assertEqual(int(x), 6)
        x = _Element(2 ** 127) * two
        self.assertEqual(int(x), 1 + 2 + 4 + 128)

    def test4(self):
        one = _Element(1)
        x = one.inverse()
        self.assertEqual(int(x), 1)
        x = _Element(82323923)
        y = x.inverse()
        self.assertEqual(int(x * y), 1)