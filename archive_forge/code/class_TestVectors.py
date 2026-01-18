import json
import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.Util.py3compat import tobytes, bchr
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128
from Cryptodome.Util.strxor import strxor
class TestVectors(unittest.TestCase):
    """Class exercising the SIV test vectors found in RFC5297"""
    test_vectors_hex = [('101112131415161718191a1b1c1d1e1f2021222324252627', '112233445566778899aabbccddee', '40c02b9690c4dc04daef7f6afe5c', '85632d07c6e8f37f950acd320a2ecc93', 'fffefdfcfbfaf9f8f7f6f5f4f3f2f1f0f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff', None), ('00112233445566778899aabbccddeeffdeaddadadeaddadaffeeddccbbaa9988' + '7766554433221100-102030405060708090a0', '7468697320697320736f6d6520706c61696e7465787420746f20656e63727970' + '74207573696e67205349562d414553', 'cb900f2fddbe404326601965c889bf17dba77ceb094fa663b7a3f748ba8af829' + 'ea64ad544a272e9c485b62a3fd5c0d', '7bdb6e3b432667eb06f4d14bff2fbd0f', '7f7e7d7c7b7a79787776757473727170404142434445464748494a4b4c4d4e4f', '09f911029d74e35bd84156c5635688c0')]
    test_vectors = [transform(tv) for tv in test_vectors_hex]

    def runTest(self):
        for assoc_data, pt, ct, mac, key, nonce in self.test_vectors:
            cipher = AES.new(key, AES.MODE_SIV, nonce=nonce)
            for x in assoc_data:
                cipher.update(x)
            ct2, mac2 = cipher.encrypt_and_digest(pt)
            self.assertEqual(ct, ct2)
            self.assertEqual(mac, mac2)
            cipher = AES.new(key, AES.MODE_SIV, nonce=nonce)
            for x in assoc_data:
                cipher.update(x)
            pt2 = cipher.decrypt_and_verify(ct, mac)
            self.assertEqual(pt, pt2)