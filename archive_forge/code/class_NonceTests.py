import unittest
from Cryptodome.Util.py3compat import bchr
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Cipher import Salsa20
from .common import make_stream_tests
class NonceTests(unittest.TestCase):

    def test_invalid_nonce_length(self):
        key = bchr(1) * 16
        self.assertRaises(ValueError, Salsa20.new, key, bchr(0) * 7)
        self.assertRaises(ValueError, Salsa20.new, key, bchr(0) * 9)

    def test_default_nonce(self):
        cipher1 = Salsa20.new(bchr(1) * 16)
        cipher2 = Salsa20.new(bchr(1) * 16)
        self.assertEqual(len(cipher1.nonce), 8)
        self.assertNotEqual(cipher1.nonce, cipher2.nonce)