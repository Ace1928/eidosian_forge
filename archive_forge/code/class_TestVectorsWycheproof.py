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
class TestVectorsWycheproof(unittest.TestCase):

    def __init__(self, wycheproof_warnings):
        unittest.TestCase.__init__(self)
        self._wycheproof_warnings = wycheproof_warnings
        self._id = 'None'

    def setUp(self):
        self.tv = []
        self.add_tests('rsa_sig_gen_misc_test.json')
        self.add_tests('rsa_signature_2048_sha224_test.json')
        self.add_tests('rsa_signature_2048_sha256_test.json')
        self.add_tests('rsa_signature_2048_sha384_test.json')
        self.add_tests('rsa_signature_2048_sha3_224_test.json')
        self.add_tests('rsa_signature_2048_sha3_256_test.json')
        self.add_tests('rsa_signature_2048_sha3_384_test.json')
        self.add_tests('rsa_signature_2048_sha3_512_test.json')
        self.add_tests('rsa_signature_2048_sha512_test.json')
        self.add_tests('rsa_signature_2048_sha512_224_test.json')
        self.add_tests('rsa_signature_2048_sha512_256_test.json')
        self.add_tests('rsa_signature_3072_sha256_test.json')
        self.add_tests('rsa_signature_3072_sha384_test.json')
        self.add_tests('rsa_signature_3072_sha3_256_test.json')
        self.add_tests('rsa_signature_3072_sha3_384_test.json')
        self.add_tests('rsa_signature_3072_sha3_512_test.json')
        self.add_tests('rsa_signature_3072_sha512_test.json')
        self.add_tests('rsa_signature_3072_sha512_256_test.json')
        self.add_tests('rsa_signature_4096_sha384_test.json')
        self.add_tests('rsa_signature_4096_sha512_test.json')
        self.add_tests('rsa_signature_4096_sha512_256_test.json')
        self.add_tests('rsa_signature_test.json')

    def add_tests(self, filename):

        def filter_rsa(group):
            return RSA.import_key(group['keyPem'])

        def filter_sha(group):
            hash_name = group['sha']
            if hash_name == 'SHA-512':
                return SHA512
            elif hash_name == 'SHA-512/224':
                return SHA512.new(truncate='224')
            elif hash_name == 'SHA-512/256':
                return SHA512.new(truncate='256')
            elif hash_name == 'SHA3-512':
                return SHA3_512
            elif hash_name == 'SHA-384':
                return SHA384
            elif hash_name == 'SHA3-384':
                return SHA3_384
            elif hash_name == 'SHA-256':
                return SHA256
            elif hash_name == 'SHA3-256':
                return SHA3_256
            elif hash_name == 'SHA-224':
                return SHA224
            elif hash_name == 'SHA3-224':
                return SHA3_224
            elif hash_name == 'SHA-1':
                return SHA1
            else:
                raise ValueError('Unknown hash algorithm: ' + hash_name)

        def filter_type(group):
            type_name = group['type']
            if type_name not in ('RsassaPkcs1Verify', 'RsassaPkcs1Generate'):
                raise ValueError('Unknown type name ' + type_name)
        result = load_test_vectors_wycheproof(('Signature', 'wycheproof'), filename, 'Wycheproof PKCS#1v1.5 signature (%s)' % filename, group_tag={'rsa_key': filter_rsa, 'hash_mod': filter_sha, 'type': filter_type})
        return result

    def shortDescription(self):
        return self._id

    def warn(self, tv):
        if tv.warning and self._wycheproof_warnings:
            import warnings
            warnings.warn('Wycheproof warning: %s (%s)' % (self._id, tv.comment))

    def test_verify(self, tv):
        self._id = 'Wycheproof RSA PKCS$#1 Test #' + str(tv.id)
        hashed_msg = tv.hash_module.new(tv.msg)
        signer = pkcs1_15.new(tv.key)
        try:
            signature = signer.verify(hashed_msg, tv.sig)
        except ValueError as e:
            if tv.warning:
                return
            assert not tv.valid
        else:
            assert tv.valid
            self.warn(tv)

    def runTest(self):
        for tv in self.tv:
            self.test_verify(tv)