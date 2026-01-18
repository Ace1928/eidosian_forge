from __future__ import print_function
import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Cryptodome.Util.py3compat import tobytes, bchr
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128, SHA256
from Cryptodome.Util.strxor import strxor
class TestVectorsGueronKrasnov(unittest.TestCase):
    """Class exercising the GCM test vectors found in
       'The fragility of AES-GCM authentication algorithm', Gueron, Krasnov
       https://eprint.iacr.org/2013/157.pdf"""

    def test_1(self):
        key = unhexlify('3da6c536d6295579c0959a7043efb503')
        iv = unhexlify('2b926197d34e091ef722db94')
        aad = unhexlify('00000000000000000000000000000000' + '000102030405060708090a0b0c0d0e0f' + '101112131415161718191a1b1c1d1e1f' + '202122232425262728292a2b2c2d2e2f' + '303132333435363738393a3b3c3d3e3f')
        digest = unhexlify('69dd586555ce3fcc89663801a71d957b')
        cipher = AES.new(key, AES.MODE_GCM, iv).update(aad)
        self.assertEqual(digest, cipher.digest())

    def test_2(self):
        key = unhexlify('843ffcf5d2b72694d19ed01d01249412')
        iv = unhexlify('dbcca32ebf9b804617c3aa9e')
        aad = unhexlify('00000000000000000000000000000000' + '101112131415161718191a1b1c1d1e1f')
        pt = unhexlify('000102030405060708090a0b0c0d0e0f' + '101112131415161718191a1b1c1d1e1f' + '202122232425262728292a2b2c2d2e2f' + '303132333435363738393a3b3c3d3e3f' + '404142434445464748494a4b4c4d4e4f')
        ct = unhexlify('6268c6fa2a80b2d137467f092f657ac0' + '4d89be2beaa623d61b5a868c8f03ff95' + 'd3dcee23ad2f1ab3a6c80eaf4b140eb0' + '5de3457f0fbc111a6b43d0763aa422a3' + '013cf1dc37fe417d1fbfc449b75d4cc5')
        digest = unhexlify('3b629ccfbc1119b7319e1dce2cd6fd6d')
        cipher = AES.new(key, AES.MODE_GCM, iv).update(aad)
        ct2, digest2 = cipher.encrypt_and_digest(pt)
        self.assertEqual(ct, ct2)
        self.assertEqual(digest, digest2)