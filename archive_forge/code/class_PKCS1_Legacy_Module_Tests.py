import json
import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import bchr
from Cryptodome.Util.number import bytes_to_long
from Cryptodome.Util.strxor import strxor
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Cryptodome.Hash import (SHA1, SHA224, SHA256, SHA384, SHA512, SHA3_384,
from Cryptodome.PublicKey import RSA
from Cryptodome.Signature import pkcs1_15
from Cryptodome.Signature import PKCS1_v1_5
from Cryptodome.Util._file_system import pycryptodome_filename
from Cryptodome.Util.strxor import strxor
class PKCS1_Legacy_Module_Tests(unittest.TestCase):
    """Verify that the legacy module Cryptodome.Signature.PKCS1_v1_5
    behaves as expected. The only difference is that the verify()
    method returns True/False and does not raise exceptions."""

    def shortDescription(self):
        return 'Test legacy Cryptodome.Signature.PKCS1_v1_5'

    def runTest(self):
        key = RSA.importKey(PKCS1_15_NoParams.rsakey)
        hashed = SHA1.new(b'Test')
        good_signature = PKCS1_v1_5.new(key).sign(hashed)
        verifier = PKCS1_v1_5.new(key.public_key())
        self.assertEqual(verifier.verify(hashed, good_signature), True)
        bad_signature = strxor(good_signature, bchr(1) * len(good_signature))
        self.assertEqual(verifier.verify(hashed, bad_signature), False)