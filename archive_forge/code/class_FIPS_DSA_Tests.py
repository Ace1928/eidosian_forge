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
class FIPS_DSA_Tests(unittest.TestCase):
    P = 118658832358899932872869585475519400247719516342448100865405466413639378222152696707107593195540012500074540023882417950871892862308677444611393983318989488754391016716002329886711216162161476818125868549160996161234237819651656260493334054804519584284232517884927253137857752483596903451871027256542552253059
    Q = 1417949633448483704686562917334218274292876272789
    G = 30330926242335176722502401833974905258077498867217200330266718029122347758449172191931128792545121758766868878943567178990352154136412275747175490355544102130000910413764989290519669244610076045319604232175663916026214948641021107450745535240348346211910194790490283450258439424429286033793042513554473025075
    X = 1126069022775112753948725909676783481321819470906
    Y = 34584118110552449252113004700350810134956261952939959513971816058999789444430242612896978114142165878190601635143632281874242490645625540667536448851860482730989520854513033347954307278872735922758009274242568828978218826147257051999542234272711232431473076347234233935320018547845428275609438074032424195851
    key_pub = DSA.construct((Y, G, P, Q))
    key_priv = DSA.construct((Y, G, P, Q, X))

    def shortDescription(self):
        return 'FIPS DSA Tests'

    def test_loopback(self):
        hashed_msg = SHA512.new(b'test')
        signer = DSS.new(self.key_priv, 'fips-186-3')
        signature = signer.sign(hashed_msg)
        verifier = DSS.new(self.key_pub, 'fips-186-3')
        verifier.verify(hashed_msg, signature)

    def test_negative_unapproved_hashes(self):
        """Verify that unapproved hashes are rejected"""
        from Cryptodome.Hash import RIPEMD160
        self.description = 'Unapproved hash (RIPEMD160) test'
        hash_obj = RIPEMD160.new()
        signer = DSS.new(self.key_priv, 'fips-186-3')
        self.assertRaises(ValueError, signer.sign, hash_obj)
        self.assertRaises(ValueError, signer.verify, hash_obj, b'\x00' * 40)

    def test_negative_unknown_modes_encodings(self):
        """Verify that unknown modes/encodings are rejected"""
        self.description = 'Unknown mode test'
        self.assertRaises(ValueError, DSS.new, self.key_priv, 'fips-186-0')
        self.description = 'Unknown encoding test'
        self.assertRaises(ValueError, DSS.new, self.key_priv, 'fips-186-3', 'xml')

    def test_asn1_encoding(self):
        """Verify ASN.1 encoding"""
        self.description = 'ASN.1 encoding test'
        hash_obj = SHA1.new()
        signer = DSS.new(self.key_priv, 'fips-186-3', 'der')
        signature = signer.sign(hash_obj)
        self.assertEqual(bord(signature[0]), 48)
        signer.verify(hash_obj, signature)
        signature = bchr(7) + signature[1:]
        self.assertRaises(ValueError, signer.verify, hash_obj, signature)

    def test_sign_verify(self):
        """Verify public/private method"""
        self.description = 'can_sign() test'
        signer = DSS.new(self.key_priv, 'fips-186-3')
        self.assertTrue(signer.can_sign())
        signer = DSS.new(self.key_pub, 'fips-186-3')
        self.assertFalse(signer.can_sign())
        try:
            signer.sign(SHA256.new(b'xyz'))
        except TypeError as e:
            msg = str(e)
        else:
            msg = ''
        self.assertTrue('Private key is needed' in msg)