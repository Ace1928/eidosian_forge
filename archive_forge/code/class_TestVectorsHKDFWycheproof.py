import re
import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import b, bchr
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Cryptodome.Hash import SHA1, HMAC, SHA256, MD5, SHA224, SHA384, SHA512
from Cryptodome.Cipher import AES, DES3
from Cryptodome.Protocol.KDF import (PBKDF1, PBKDF2, _S2V, HKDF, scrypt,
from Cryptodome.Protocol.KDF import _bcrypt_decode
class TestVectorsHKDFWycheproof(unittest.TestCase):

    def __init__(self, wycheproof_warnings):
        unittest.TestCase.__init__(self)
        self._wycheproof_warnings = wycheproof_warnings
        self._id = 'None'

    def add_tests(self, filename):

        def filter_algo(root):
            algo_name = root['algorithm']
            if algo_name == 'HKDF-SHA-1':
                return SHA1
            elif algo_name == 'HKDF-SHA-256':
                return SHA256
            elif algo_name == 'HKDF-SHA-384':
                return SHA384
            elif algo_name == 'HKDF-SHA-512':
                return SHA512
            else:
                raise ValueError('Unknown algorithm ' + algo_name)

        def filter_size(unit):
            return int(unit['size'])
        result = load_test_vectors_wycheproof(('Protocol', 'wycheproof'), filename, 'Wycheproof HMAC (%s)' % filename, root_tag={'hash_module': filter_algo}, unit_tag={'size': filter_size})
        return result

    def setUp(self):
        self.tv = []
        self.add_tests('hkdf_sha1_test.json')
        self.add_tests('hkdf_sha256_test.json')
        self.add_tests('hkdf_sha384_test.json')
        self.add_tests('hkdf_sha512_test.json')

    def shortDescription(self):
        return self._id

    def warn(self, tv):
        if tv.warning and self._wycheproof_warnings:
            import warnings
            warnings.warn('Wycheproof warning: %s (%s)' % (self._id, tv.comment))

    def test_verify(self, tv):
        self._id = 'Wycheproof HKDF Test #%d (%s, %s)' % (tv.id, tv.comment, tv.filename)
        try:
            key = HKDF(tv.ikm, tv.size, tv.salt, tv.hash_module, 1, tv.info)
        except ValueError:
            assert not tv.valid
        else:
            if key != tv.okm:
                assert not tv.valid
            else:
                assert tv.valid
                self.warn(tv)

    def runTest(self):
        for tv in self.tv:
            self.test_verify(tv)