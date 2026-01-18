import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.PublicKey import ECC
from Cryptodome.PublicKey.ECC import EccPoint, _curves, EccKey
from Cryptodome.Math.Numbers import Integer
class TestEccKey_P256(unittest.TestCase):

    def test_private_key(self):
        key = EccKey(curve='P-256', d=1)
        self.assertEqual(key.d, 1)
        self.assertTrue(key.has_private())
        self.assertEqual(key.pointQ.x, _curves['p256'].Gx)
        self.assertEqual(key.pointQ.y, _curves['p256'].Gy)
        point = EccPoint(_curves['p256'].Gx, _curves['p256'].Gy)
        key = EccKey(curve='P-256', d=1, point=point)
        self.assertEqual(key.d, 1)
        self.assertTrue(key.has_private())
        self.assertEqual(key.pointQ, point)
        key = EccKey(curve='secp256r1', d=1)
        key = EccKey(curve='prime256v1', d=1)
        self.assertRaises(ValueError, EccKey, curve='p256', seed=b'H' * 32)

    def test_public_key(self):
        point = EccPoint(_curves['p256'].Gx, _curves['p256'].Gy)
        key = EccKey(curve='P-256', point=point)
        self.assertFalse(key.has_private())
        self.assertEqual(key.pointQ, point)

    def test_public_key_derived(self):
        priv_key = EccKey(curve='P-256', d=3)
        pub_key = priv_key.public_key()
        self.assertFalse(pub_key.has_private())
        self.assertEqual(priv_key.pointQ, pub_key.pointQ)

    def test_invalid_curve(self):
        self.assertRaises(ValueError, lambda: EccKey(curve='P-257', d=1))

    def test_invalid_d(self):
        self.assertRaises(ValueError, lambda: EccKey(curve='P-256', d=0))
        self.assertRaises(ValueError, lambda: EccKey(curve='P-256', d=_curves['p256'].order))

    def test_equality(self):
        private_key = ECC.construct(d=3, curve='P-256')
        private_key2 = ECC.construct(d=3, curve='P-256')
        private_key3 = ECC.construct(d=4, curve='P-256')
        public_key = private_key.public_key()
        public_key2 = private_key2.public_key()
        public_key3 = private_key3.public_key()
        self.assertEqual(private_key, private_key2)
        self.assertNotEqual(private_key, private_key3)
        self.assertEqual(public_key, public_key2)
        self.assertNotEqual(public_key, public_key3)
        self.assertNotEqual(public_key, private_key)

    def test_name_consistency(self):
        key = ECC.generate(curve='p256')
        self.assertIn("curve='NIST P-256'", repr(key))
        self.assertEqual(key.curve, 'NIST P-256')
        self.assertEqual(key.public_key().curve, 'NIST P-256')