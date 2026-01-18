import unittest
from binascii import unhexlify as uh
from Cryptodome.Util.py3compat import *
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.Padding import pad, unpad
class ISO7816_Tests(unittest.TestCase):

    def test1(self):
        padded = pad(b(''), 4, 'iso7816')
        self.assertTrue(padded == uh(b('80000000')))
        back = unpad(padded, 4, 'iso7816')
        self.assertTrue(back == b(''))

    def test2(self):
        padded = pad(uh(b('12345678')), 4, 'iso7816')
        self.assertTrue(padded == uh(b('1234567880000000')))
        back = unpad(padded, 4, 'iso7816')
        self.assertTrue(back == uh(b('12345678')))

    def test3(self):
        padded = pad(uh(b('123456')), 4, 'iso7816')
        self.assertTrue(padded == uh(b('12345680')))
        back = unpad(padded, 4, 'iso7816')
        self.assertTrue(back == uh(b('123456')))

    def test4(self):
        padded = pad(uh(b('1234567890')), 4, 'iso7816')
        self.assertTrue(padded == uh(b('1234567890800000')))
        back = unpad(padded, 4, 'iso7816')
        self.assertTrue(back == uh(b('1234567890')))

    def testn1(self):
        self.assertRaises(ValueError, unpad, b('123456\x81'), 4, 'iso7816')
        self.assertRaises(ValueError, unpad, b(''), 4, 'iso7816')