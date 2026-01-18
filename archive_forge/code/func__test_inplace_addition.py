import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.PublicKey import ECC
from Cryptodome.PublicKey.ECC import EccPoint, _curves, EccKey
from Cryptodome.Math.Numbers import Integer
def _test_inplace_addition(self):
    pointRx = 51876823396606567385074157065323717879096505862684305408641575318717059434110
    pointRy = 63932234198967235656258550897228394755510754316810467717045609836559811134052
    pointR = self.pointS.copy()
    pointR += self.pointT
    self.assertEqual(pointR.x, pointRx)
    self.assertEqual(pointR.y, pointRy)
    pai = pointR.point_at_infinity()
    pointR = self.pointS.copy()
    pointR += pai
    self.assertEqual(pointR, self.pointS)
    pointR = pai.copy()
    pointR += self.pointS
    self.assertEqual(pointR, self.pointS)
    pointR = pai.copy()
    pointR += pai
    self.assertEqual(pointR, pai)