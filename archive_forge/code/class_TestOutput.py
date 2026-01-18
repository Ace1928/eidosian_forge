from __future__ import print_function
import unittest
from Cryptodome.Hash import SHA256
from Cryptodome.Cipher import AES
from Cryptodome.Util.py3compat import *
from binascii import hexlify
class TestOutput(unittest.TestCase):

    def __init__(self, use_aesni):
        unittest.TestCase.__init__(self)
        self.use_aesni = use_aesni

    def runTest(self):
        cipher = AES.new(b'4' * 16, AES.MODE_ECB, use_aesni=self.use_aesni)
        pt = b'5' * 16
        ct = cipher.encrypt(pt)
        output = bytearray(16)
        res = cipher.encrypt(pt, output=output)
        self.assertEqual(ct, output)
        self.assertEqual(res, None)
        res = cipher.decrypt(ct, output=output)
        self.assertEqual(pt, output)
        self.assertEqual(res, None)
        output = memoryview(bytearray(16))
        cipher.encrypt(pt, output=output)
        self.assertEqual(ct, output)
        cipher.decrypt(ct, output=output)
        self.assertEqual(pt, output)
        self.assertRaises(TypeError, cipher.encrypt, pt, output=b'0' * 16)
        self.assertRaises(TypeError, cipher.decrypt, ct, output=b'0' * 16)
        shorter_output = bytearray(15)
        self.assertRaises(ValueError, cipher.encrypt, pt, output=shorter_output)
        self.assertRaises(ValueError, cipher.decrypt, ct, output=shorter_output)