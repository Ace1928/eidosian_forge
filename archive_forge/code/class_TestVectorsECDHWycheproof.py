import re
import unittest
from binascii import hexlify
from Cryptodome.Util.py3compat import bord
from Cryptodome.Hash import SHA256
from Cryptodome.PublicKey import ECC
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Cryptodome.Protocol.DH import key_agreement
class TestVectorsECDHWycheproof(unittest.TestCase):
    desc = 'Wycheproof ECDH tests'

    def add_tests(self, filename):

        def curve(g):
            return g['curve']

        def private(u):
            return int(u['private'], 16)
        result = load_test_vectors_wycheproof(('Protocol', 'wycheproof'), filename, 'Wycheproof ECDH (%s)' % filename, group_tag={'curve': curve}, unit_tag={'private': private})
        self.tv += result

    def setUp(self):
        self.tv = []
        self.desc = None
        self.add_tests('ecdh_secp224r1_ecpoint_test.json')
        self.add_tests('ecdh_secp256r1_ecpoint_test.json')
        self.add_tests('ecdh_secp384r1_ecpoint_test.json')
        self.add_tests('ecdh_secp521r1_ecpoint_test.json')
        self.add_tests('ecdh_secp224r1_test.json')
        self.add_tests('ecdh_secp256r1_test.json')
        self.add_tests('ecdh_secp384r1_test.json')
        self.add_tests('ecdh_secp521r1_test.json')

    def shortDescription(self):
        return self.desc

    def test_verify(self, tv):
        if len(tv.public) == 0:
            return
        try:
            if bord(tv.public[0]) == 4:
                public_key = ECC.import_key(tv.public, curve_name=tv.curve)
            else:
                public_key = ECC.import_key(tv.public)
        except ValueError:
            assert tv.warning or not tv.valid
            return
        private_key = ECC.construct(curve=tv.curve, d=tv.private)
        try:
            z = key_agreement(static_pub=public_key, static_priv=private_key, kdf=lambda x: x)
        except ValueError:
            assert not tv.valid
        except TypeError as e:
            assert not tv.valid
            assert 'incompatible curve' in str(e)
        else:
            self.assertEqual(z, tv.shared)
            assert tv.valid

    def runTest(self):
        for tv in self.tv:
            self.desc = 'Wycheproof ECDH Verify Test #%d (%s, %s)' % (tv.id, tv.comment, tv.filename)
            self.test_verify(tv)