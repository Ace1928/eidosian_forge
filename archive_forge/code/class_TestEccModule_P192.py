import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.PublicKey import ECC
from Cryptodome.PublicKey.ECC import EccPoint, _curves, EccKey
from Cryptodome.Math.Numbers import Integer
class TestEccModule_P192(unittest.TestCase):

    def test_generate(self):
        key = ECC.generate(curve='P-192')
        self.assertTrue(key.has_private())
        self.assertEqual(key.pointQ, EccPoint(_curves['p192'].Gx, _curves['p192'].Gy, 'P-192') * key.d, 'p192')
        ECC.generate(curve='secp192r1')
        ECC.generate(curve='prime192v1')

    def test_construct(self):
        key = ECC.construct(curve='P-192', d=1)
        self.assertTrue(key.has_private())
        self.assertEqual(key.pointQ, _curves['p192'].G)
        key = ECC.construct(curve='P-192', point_x=_curves['p192'].Gx, point_y=_curves['p192'].Gy)
        self.assertFalse(key.has_private())
        self.assertEqual(key.pointQ, _curves['p192'].G)
        ECC.construct(curve='p192', d=1)
        ECC.construct(curve='secp192r1', d=1)
        ECC.construct(curve='prime192v1', d=1)

    def test_negative_construct(self):
        coord = dict(point_x=10, point_y=4)
        coordG = dict(point_x=_curves['p192'].Gx, point_y=_curves['p192'].Gy)
        self.assertRaises(ValueError, ECC.construct, curve='P-192', **coord)
        self.assertRaises(ValueError, ECC.construct, curve='P-192', d=2, **coordG)