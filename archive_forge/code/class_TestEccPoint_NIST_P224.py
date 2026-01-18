import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.PublicKey import ECC
from Cryptodome.PublicKey.ECC import EccPoint, _curves, EccKey
from Cryptodome.Math.Numbers import Integer
class TestEccPoint_NIST_P224(unittest.TestCase):
    """Tests defined in section 4.2 of https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.204.9073&rep=rep1&type=pdf"""
    pointS = EccPoint(11667657833536944514422240901140695971851802309747941616875054611291, 25200609023134323751445127690242957618717340442338763601367545659170, curve='p224')
    pointT = EccPoint(19289899102325740925706901573554401888796650834671847859774556650861, 20658709206965647193440449148963768291283864582375558978411284204596, curve='p224')

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
        pointRx = 3731655391337843992307582674158418281542776652333455233821634133269
        pointRy = 24141506944378136799012218316934199199165503396763489643549684652656
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
        pointRx = 3731655391337843992307582674158418281542776652333455233821634133269
        pointRy = 24141506944378136799012218316934199199165503396763489643549684652656
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
        pointRx = 17880642473844131068660536455251697562005543173265332392907270578119
        pointRy = 18318393913229721844211669900027039248755787144076075895349737151869
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
        d = 17645073490574221494934459378684546147142719135929893683404110681915
        pointRx = 15865701639123720156335339805876942712784874763719651919734220438692
        pointRy = 1638239606610309445617153230568115391638678751727177140440968457488
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

    def test_joing_scalar_multiply(self):
        d = 17645073490574221494934459378684546147142719135929893683404110681915
        e = 8933974529695623223734491032809274731595579280135540257298545071926
        pointRx = 23167947856545279553341884159008667816851636417286522497550711834686
        pointRy = 4983454765697875611697939615323943451605610211599671255036723966347
        pointR = self.pointS * d + self.pointT * e
        self.assertEqual(pointR.x, pointRx)
        self.assertEqual(pointR.y, pointRy)

    def test_sizes(self):
        self.assertEqual(self.pointS.size_in_bits(), 224)
        self.assertEqual(self.pointS.size_in_bytes(), 28)