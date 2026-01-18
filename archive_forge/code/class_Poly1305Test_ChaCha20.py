import json
import unittest
from binascii import unhexlify, hexlify
from .common import make_mac_tests
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import Poly1305
from Cryptodome.Cipher import AES, ChaCha20
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.Util.strxor import strxor_c
class Poly1305Test_ChaCha20(unittest.TestCase):
    key = b'\x11' * 32

    def test_new_positive(self):
        data = b'r' * 100
        h1 = Poly1305.new(key=self.key, cipher=ChaCha20)
        self.assertEqual(h1.digest_size, 16)
        self.assertEqual(len(h1.nonce), 12)
        h2 = Poly1305.new(key=self.key, cipher=ChaCha20, nonce=b'8' * 8)
        self.assertEqual(len(h2.nonce), 8)
        self.assertEqual(h2.nonce, b'8' * 8)

    def test_new_negative(self):
        self.assertRaises(ValueError, Poly1305.new, key=self.key, nonce=b'1' * 7, cipher=ChaCha20)