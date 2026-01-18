import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.PublicKey import ECC
from Cryptodome.PublicKey.ECC import EccPoint, _curves, EccKey
from Cryptodome.Math.Numbers import Integer
class TestEccPoint_NIST_P192(unittest.TestCase):
    """Tests defined in section 4.1 of https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.204.9073&rep=rep1&type=pdf"""
    pointS = EccPoint(5206740333133064750202129362262409028415794571159000296696, 1234539015068213699179346148240981644175632721097521553211, curve='p192')
    pointT = EccPoint(5938062382599723486047778306805458657136056379418357486500, 938283877858310256697963661438335331434554539784311315577, curve='p192')

    def test_set(self):
        pointW = EccPoint(0, 0)
        pointW.set(self.pointS)
        self.assertEqual(pointW, self.pointS)

    def test_copy(self):
        pointW = self.pointS.copy()
        self.assertEqual(pointW, self.pointS)
        pointW.set(self.pointT)
        self.assertEqual(pointW, self.pointT)
        self.assertNotEqual(self.pointS, self.pointT)

    def test_negate(self):
        negS = -self.pointS
        sum = self.pointS + negS
        self.assertEqual(sum, self.pointS.point_at_infinity())

    def test_addition(self):
        pointRx = 1787070900316344022479363585363935252075532448940096815760
        pointRy = 1583034776780933252095415958625802984888372377603917916747
        pointR = self.pointS + self.pointT
        self.assertEqual(pointR.x, pointRx)
        self.assertEqual(pointR.y, pointRy)
        pai = pointR.point_at_infinity()
        pointR = self.pointS + pai
        self.assertEqual(pointR, self.pointS)
        pointR = pai + self.pointS
        self.assertEqual(pointR, self.pointS)
        pointR = pai + pai
        self.assertEqual(pointR, pai)

    def test_inplace_addition(self):
        pointRx = 1787070900316344022479363585363935252075532448940096815760
        pointRy = 1583034776780933252095415958625802984888372377603917916747
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

    def test_doubling(self):
        pointRx = 1195895923065450997501505402941681398272052708885411031394
        pointRy = 340030206158745947396451508065335698335058477174385838543
        pointR = self.pointS.copy()
        pointR.double()
        self.assertEqual(pointR.x, pointRx)
        self.assertEqual(pointR.y, pointRy)
        pai = self.pointS.point_at_infinity()
        pointR = pai.copy()
        pointR.double()
        self.assertEqual(pointR, pai)
        pointR = self.pointS.copy()
        pointR += pointR
        self.assertEqual(pointR.x, pointRx)
        self.assertEqual(pointR.y, pointRy)

    def test_scalar_multiply(self):
        d = 4108059114144203890209012299831570373806242553631966262510
        pointRx = 776869029487597406993006453288107978891554090426994486577
        pointRy = 2352649282612884123934831611790818939149682285969215126278
        pointR = self.pointS * d
        self.assertEqual(pointR.x, pointRx)
        self.assertEqual(pointR.y, pointRy)
        pai = self.pointS.point_at_infinity()
        pointR = self.pointS * 0
        self.assertEqual(pointR, pai)
        self.assertRaises(ValueError, lambda: self.pointS * -1)
        pointR = d * self.pointS
        self.assertEqual(pointR.x, pointRx)
        self.assertEqual(pointR.y, pointRy)
        pointR = Integer(d) * self.pointS
        self.assertEqual(pointR.x, pointRx)
        self.assertEqual(pointR.y, pointRy)

    def test_joint_scalar_multiply(self):
        d = 4108059114144203890209012299831570373806242553631966262510
        e = 4824127346165818618562330903884116653311202509745852871356
        pointRx = 39786866609245082371772779541859439402855864496422607838
        pointRy = 547967566579883709478937502153554894699060378424501614148
        pointR = self.pointS * d + self.pointT * e
        self.assertEqual(pointR.x, pointRx)
        self.assertEqual(pointR.y, pointRy)

    def test_sizes(self):
        self.assertEqual(self.pointS.size_in_bits(), 192)
        self.assertEqual(self.pointS.size_in_bytes(), 24)