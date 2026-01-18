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
class PKCS1_All_Hashes_Tests(unittest.TestCase):

    def shortDescription(self):
        return 'Test PKCS#1v1.5 signature in combination with all hashes'

    def runTest(self):
        key = RSA.generate(1024)
        signer = pkcs1_15.new(key)
        hash_names = ('MD2', 'MD4', 'MD5', 'RIPEMD160', 'SHA1', 'SHA224', 'SHA256', 'SHA384', 'SHA512', 'SHA3_224', 'SHA3_256', 'SHA3_384', 'SHA3_512')
        for name in hash_names:
            hashed = load_hash_by_name(name).new(b'Test')
            signer.sign(hashed)
        from Cryptodome.Hash import BLAKE2b, BLAKE2s
        for hash_size in (20, 32, 48, 64):
            hashed_b = BLAKE2b.new(digest_bytes=hash_size, data=b'Test')
            signer.sign(hashed_b)
        for hash_size in (16, 20, 28, 32):
            hashed_s = BLAKE2s.new(digest_bytes=hash_size, data=b'Test')
            signer.sign(hashed_s)