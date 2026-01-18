import json
import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.Hash import CMAC
from Cryptodome.Cipher import AES, DES3
from Cryptodome.Hash import SHAKE128
from Cryptodome.Util.strxor import strxor
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
class TestCMAC(unittest.TestCase):

    def test_internal_caching(self):
        """Verify that internal caching is implemented correctly"""
        data_to_mac = get_tag_random('data_to_mac', 128)
        key = get_tag_random('key', 16)
        ref_mac = CMAC.new(key, msg=data_to_mac, ciphermod=AES).digest()
        for chunk_length in (1, 2, 3, 7, 10, 13, 16, 40, 80, 128):
            chunks = [data_to_mac[i:i + chunk_length] for i in range(0, len(data_to_mac), chunk_length)]
            mac = CMAC.new(key, ciphermod=AES)
            for chunk in chunks:
                mac.update(chunk)
            self.assertEqual(ref_mac, mac.digest())

    def test_update_after_digest(self):
        msg = b'rrrrttt'
        key = b'4' * 16
        h = CMAC.new(key, msg[:4], ciphermod=AES)
        dig1 = h.digest()
        self.assertRaises(TypeError, h.update, msg[4:])
        dig2 = CMAC.new(key, msg, ciphermod=AES).digest()
        h2 = CMAC.new(key, msg[:4], ciphermod=AES, update_after_digest=True)
        self.assertEqual(h2.digest(), dig1)
        h2.update(msg[4:])
        self.assertEqual(h2.digest(), dig2)