import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.PublicKey import ECC
from Cryptodome.PublicKey.ECC import EccPoint, _curves, EccKey
from Cryptodome.Math.Numbers import Integer
class TestEccKey_P384(unittest.TestCase):

    def test_private_key(self):
        p384 = _curves['p384']
        key = EccKey(curve='P-384', d=1)
        self.assertEqual(key.d, 1)
        self.assertTrue(key.has_private())
        self.assertEqual(key.pointQ.x, p384.Gx)
        self.assertEqual(key.pointQ.y, p384.Gy)
        point = EccPoint(p384.Gx, p384.Gy, 'p384')
        key = EccKey(curve='P-384', d=1, point=point)
        self.assertEqual(key.d, 1)
        self.assertTrue(key.has_private())
        self.assertEqual(key.pointQ, point)
        key = EccKey(curve='p384', d=1)
        key = EccKey(curve='secp384r1', d=1)
        key = EccKey(curve='prime384v1', d=1)

    def test_public_key(self):
        p384 = _curves['p384']
        point = EccPoint(p384.Gx, p384.Gy, 'p384')
        key = EccKey(curve='P-384', point=point)
        self.assertFalse(key.has_private())
        self.assertEqual(key.pointQ, point)

    def test_public_key_derived(self):
        priv_key = EccKey(curve='P-384', d=3)
        pub_key = priv_key.public_key()
        self.assertFalse(pub_key.has_private())
        self.assertEqual(priv_key.pointQ, pub_key.pointQ)

    def test_invalid_curve(self):
        self.assertRaises(ValueError, lambda: EccKey(curve='P-385', d=1))

    def test_invalid_d(self):
        self.assertRaises(ValueError, lambda: EccKey(curve='P-384', d=0))
        self.assertRaises(ValueError, lambda: EccKey(curve='P-384', d=_curves['p384'].order))

    def test_equality(self):
        private_key = ECC.construct(d=3, curve='P-384')
        private_key2 = ECC.construct(d=3, curve='P-384')
        private_key3 = ECC.construct(d=4, curve='P-384')
        public_key = private_key.public_key()
        public_key2 = private_key2.public_key()
        public_key3 = private_key3.public_key()
        self.assertEqual(private_key, private_key2)
        self.assertNotEqual(private_key, private_key3)
        self.assertEqual(public_key, public_key2)
        self.assertNotEqual(public_key, public_key3)
        self.assertNotEqual(public_key, private_key)

    def test_name_consistency(self):
        key = ECC.generate(curve='p384')
        self.assertIn("curve='NIST P-384'", repr(key))
        self.assertEqual(key.curve, 'NIST P-384')
        self.assertEqual(key.public_key().curve, 'NIST P-384')