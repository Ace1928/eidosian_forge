import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.PublicKey import ECC
from Cryptodome.PublicKey.ECC import EccPoint, _curves, EccKey
from Cryptodome.Math.Numbers import Integer
class TestEccModule_P384(unittest.TestCase):

    def test_generate(self):
        curve = _curves['p384']
        key = ECC.generate(curve='P-384')
        self.assertTrue(key.has_private())
        self.assertEqual(key.pointQ, EccPoint(curve.Gx, curve.Gy, 'p384') * key.d)
        ECC.generate(curve='secp384r1')
        ECC.generate(curve='prime384v1')

    def test_construct(self):
        curve = _curves['p384']
        key = ECC.construct(curve='P-384', d=1)
        self.assertTrue(key.has_private())
        self.assertEqual(key.pointQ, _curves['p384'].G)
        key = ECC.construct(curve='P-384', point_x=curve.Gx, point_y=curve.Gy)
        self.assertFalse(key.has_private())
        self.assertEqual(key.pointQ, curve.G)
        ECC.construct(curve='p384', d=1)
        ECC.construct(curve='secp384r1', d=1)
        ECC.construct(curve='prime384v1', d=1)

    def test_negative_construct(self):
        coord = dict(point_x=10, point_y=4)
        coordG = dict(point_x=_curves['p384'].Gx, point_y=_curves['p384'].Gy)
        self.assertRaises(ValueError, ECC.construct, curve='P-384', **coord)
        self.assertRaises(ValueError, ECC.construct, curve='P-384', d=2, **coordG)