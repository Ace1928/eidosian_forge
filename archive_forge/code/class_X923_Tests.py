import unittest
from binascii import unhexlify as uh
from Cryptodome.Util.py3compat import *
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.Padding import pad, unpad
class X923_Tests(unittest.TestCase):

    def test1(self):
        padded = pad(b(''), 4, 'x923')
        self.assertTrue(padded == uh(b('00000004')))
        back = unpad(padded, 4, 'x923')
        self.assertTrue(back == b(''))

    def test2(self):
        padded = pad(uh(b('12345678')), 4, 'x923')
        self.assertTrue(padded == uh(b('1234567800000004')))
        back = unpad(padded, 4, 'x923')
        self.assertTrue(back == uh(b('12345678')))

    def test3(self):
        padded = pad(uh(b('123456')), 4, 'x923')
        self.assertTrue(padded == uh(b('12345601')))
        back = unpad(padded, 4, 'x923')
        self.assertTrue(back == uh(b('123456')))

    def test4(self):
        padded = pad(uh(b('1234567890')), 4, 'x923')
        self.assertTrue(padded == uh(b('1234567890000003')))
        back = unpad(padded, 4, 'x923')
        self.assertTrue(back == uh(b('1234567890')))

    def testn1(self):
        self.assertRaises(ValueError, unpad, b('123456\x02'), 4, 'x923')
        self.assertRaises(ValueError, unpad, b('123456\x00'), 4, 'x923')
        self.assertRaises(ValueError, unpad, b('123456\x00\x00\x00\x00\x05'), 4, 'x923')
        self.assertRaises(ValueError, unpad, b(''), 4, 'x923')