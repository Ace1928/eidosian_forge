import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.PublicKey import ECC
from Cryptodome.PublicKey.ECC import EccPoint, _curves, EccKey
from Cryptodome.Math.Numbers import Integer
class TestEccModule_P224(unittest.TestCase):

    def test_generate(self):
        key = ECC.generate(curve='P-224')
        self.assertTrue(key.has_private())
        self.assertEqual(key.pointQ, EccPoint(_curves['p224'].Gx, _curves['p224'].Gy, 'P-224') * key.d, 'p224')
        ECC.generate(curve='secp224r1')
        ECC.generate(curve='prime224v1')

    def test_construct(self):
        key = ECC.construct(curve='P-224', d=1)
        self.assertTrue(key.has_private())
        self.assertEqual(key.pointQ, _curves['p224'].G)
        key = ECC.construct(curve='P-224', point_x=_curves['p224'].Gx, point_y=_curves['p224'].Gy)
        self.assertFalse(key.has_private())
        self.assertEqual(key.pointQ, _curves['p224'].G)
        ECC.construct(curve='p224', d=1)
        ECC.construct(curve='secp224r1', d=1)
        ECC.construct(curve='prime224v1', d=1)

    def test_negative_construct(self):
        coord = dict(point_x=10, point_y=4)
        coordG = dict(point_x=_curves['p224'].Gx, point_y=_curves['p224'].Gy)
        self.assertRaises(ValueError, ECC.construct, curve='P-224', **coord)
        self.assertRaises(ValueError, ECC.construct, curve='P-224', d=2, **coordG)