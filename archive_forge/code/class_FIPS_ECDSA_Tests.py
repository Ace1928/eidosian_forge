import re
import unittest
from binascii import hexlify, unhexlify
from Cryptodome.Util.py3compat import tobytes, bord, bchr
from Cryptodome.Hash import (SHA1, SHA224, SHA256, SHA384, SHA512,
from Cryptodome.Signature import DSS
from Cryptodome.PublicKey import DSA, ECC
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Cryptodome.Util.number import bytes_to_long, long_to_bytes
class FIPS_ECDSA_Tests(unittest.TestCase):
    key_priv = ECC.generate(curve='P-256')
    key_pub = key_priv.public_key()

    def shortDescription(self):
        return 'FIPS ECDSA Tests'

    def test_loopback(self):
        hashed_msg = SHA512.new(b'test')
        signer = DSS.new(self.key_priv, 'fips-186-3')
        signature = signer.sign(hashed_msg)
        verifier = DSS.new(self.key_pub, 'fips-186-3')
        verifier.verify(hashed_msg, signature)

    def test_negative_unapproved_hashes(self):
        """Verify that unapproved hashes are rejected"""
        from Cryptodome.Hash import SHA1
        self.description = 'Unapproved hash (SHA-1) test'
        hash_obj = SHA1.new()
        signer = DSS.new(self.key_priv, 'fips-186-3')
        self.assertRaises(ValueError, signer.sign, hash_obj)
        self.assertRaises(ValueError, signer.verify, hash_obj, b'\x00' * 40)

    def test_negative_eddsa_key(self):
        key = ECC.generate(curve='ed25519')
        self.assertRaises(ValueError, DSS.new, key, 'fips-186-3')

    def test_sign_verify(self):
        """Verify public/private method"""
        self.description = 'can_sign() test'
        signer = DSS.new(self.key_priv, 'fips-186-3')
        self.assertTrue(signer.can_sign())
        signer = DSS.new(self.key_pub, 'fips-186-3')
        self.assertFalse(signer.can_sign())
        self.assertRaises(TypeError, signer.sign, SHA256.new(b'xyz'))
        try:
            signer.sign(SHA256.new(b'xyz'))
        except TypeError as e:
            msg = str(e)
        else:
            msg = ''
        self.assertTrue('Private key is needed' in msg)

    def test_negative_unknown_modes_encodings(self):
        """Verify that unknown modes/encodings are rejected"""
        self.description = 'Unknown mode test'
        self.assertRaises(ValueError, DSS.new, self.key_priv, 'fips-186-0')
        self.description = 'Unknown encoding test'
        self.assertRaises(ValueError, DSS.new, self.key_priv, 'fips-186-3', 'xml')

    def test_asn1_encoding(self):
        """Verify ASN.1 encoding"""
        self.description = 'ASN.1 encoding test'
        hash_obj = SHA256.new()
        signer = DSS.new(self.key_priv, 'fips-186-3', 'der')
        signature = signer.sign(hash_obj)
        self.assertEqual(bord(signature[0]), 48)
        signer.verify(hash_obj, signature)
        signature = bchr(7) + signature[1:]
        self.assertRaises(ValueError, signer.verify, hash_obj, signature)