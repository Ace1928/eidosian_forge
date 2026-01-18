import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.PublicKey import ECC
from Cryptodome.PublicKey.ECC import EccPoint, _curves, EccKey
from Cryptodome.Math.Numbers import Integer
class TestEccModule_P256(unittest.TestCase):

    def test_generate(self):
        key = ECC.generate(curve='P-256')
        self.assertTrue(key.has_private())
        self.assertEqual(key.pointQ, EccPoint(_curves['p256'].Gx, _curves['p256'].Gy) * key.d, 'p256')
        ECC.generate(curve='secp256r1')
        ECC.generate(curve='prime256v1')

    def test_construct(self):
        key = ECC.construct(curve='P-256', d=1)
        self.assertTrue(key.has_private())
        self.assertEqual(key.pointQ, _curves['p256'].G)
        key = ECC.construct(curve='P-256', point_x=_curves['p256'].Gx, point_y=_curves['p256'].Gy)
        self.assertFalse(key.has_private())
        self.assertEqual(key.pointQ, _curves['p256'].G)
        ECC.construct(curve='p256', d=1)
        ECC.construct(curve='secp256r1', d=1)
        ECC.construct(curve='prime256v1', d=1)

    def test_negative_construct(self):
        coord = dict(point_x=10, point_y=4)
        coordG = dict(point_x=_curves['p256'].Gx, point_y=_curves['p256'].Gy)
        self.assertRaises(ValueError, ECC.construct, curve='P-256', **coord)
        self.assertRaises(ValueError, ECC.construct, curve='P-256', d=2, **coordG)