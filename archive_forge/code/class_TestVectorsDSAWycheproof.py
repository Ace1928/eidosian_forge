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
class TestVectorsDSAWycheproof(unittest.TestCase):

    def __init__(self, wycheproof_warnings, slow_tests):
        unittest.TestCase.__init__(self)
        self._wycheproof_warnings = wycheproof_warnings
        self._slow_tests = slow_tests
        self._id = 'None'
        self.tv = []

    def setUp(self):

        def filter_dsa(group):
            return DSA.import_key(group['keyPem'])

        def filter_sha(group):
            return get_hash_module(group['sha'])

        def filter_type(group):
            sig_type = group['type']
            if sig_type != 'DsaVerify':
                raise ValueError('Unknown signature type ' + sig_type)
            return sig_type
        result = load_test_vectors_wycheproof(('Signature', 'wycheproof'), 'dsa_test.json', 'Wycheproof DSA signature', group_tag={'key': filter_dsa, 'hash_module': filter_sha, 'sig_type': filter_type})
        self.tv += result

    def shortDescription(self):
        return self._id

    def warn(self, tv):
        if tv.warning and self._wycheproof_warnings:
            import warnings
            warnings.warn('Wycheproof warning: %s (%s)' % (self._id, tv.comment))

    def test_verify(self, tv):
        self._id = 'Wycheproof DSA Test #' + str(tv.id)
        hashed_msg = tv.hash_module.new(tv.msg)
        signer = DSS.new(tv.key, 'fips-186-3', encoding='der')
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