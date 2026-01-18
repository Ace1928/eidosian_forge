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
class FIPS_PKCS1_Verify_Tests(unittest.TestCase):

    def shortDescription(self):
        return 'FIPS PKCS1 Tests (Verify)'

    def test_can_sign(self):
        test_public_key = RSA.generate(1024).public_key()
        verifier = pkcs1_15.new(test_public_key)
        self.assertEqual(verifier.can_sign(), False)