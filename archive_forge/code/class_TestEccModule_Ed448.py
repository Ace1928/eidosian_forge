import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.PublicKey import ECC
from Cryptodome.PublicKey.ECC import EccPoint, _curves, EccKey
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Hash import SHAKE128
class TestEccModule_Ed448(unittest.TestCase):

    def test_generate(self):
        key = ECC.generate(curve='Ed448')
        self.assertTrue(key.has_private())
        point = EccPoint(_curves['Ed448'].Gx, _curves['Ed448'].Gy, curve='Ed448') * key.d
        self.assertEqual(key.pointQ, point)
        key2 = ECC.generate(curve='Ed448')
        self.assertNotEqual(key, key2)
        ECC.generate(curve='Ed448')
        key1 = ECC.generate(curve='Ed448', randfunc=SHAKE128.new().read)
        key2 = ECC.generate(curve='Ed448', randfunc=SHAKE128.new().read)
        self.assertEqual(key1, key2)

    def test_construct(self):
        seed = unhexlify('4adf5d37ac6785e83e99a924f92676d366a78690af59c92b6bdf14f9cdbcf26fdad478109607583d633b60078d61d51d81b7509c5433b0d4c9')
        Px = 325446217306090953437856287370448806488502865468978757199471133175786932082059105257667529930261589134008879049952195686473111433291287
        Py = 448740337158713015934986081827345765524730480111986550601798045359153740662802087188188588374402113723130746791122877945925962315214529
        d = 501087328496366969140677053865823318065705212104179624634307461836797857615836433313593441736527960295998751889288610349510870818185088
        point = EccPoint(Px, Py, curve='Ed448')
        key = ECC.construct(curve='Ed448', seed=seed)
        self.assertEqual(key.pointQ, point)
        self.assertTrue(key.has_private())
        key = ECC.construct(curve='Ed448', point_x=Px, point_y=Py)
        self.assertEqual(key.pointQ, point)
        self.assertFalse(key.has_private())
        key = ECC.construct(curve='Ed448', seed=seed, point_x=Px, point_y=Py)
        self.assertEqual(key.pointQ, point)
        self.assertTrue(key.has_private())
        key = ECC.construct(curve='ed448', seed=seed)

    def test_negative_construct(self):
        coord = dict(point_x=10, point_y=4)
        coordG = dict(point_x=_curves['ed448'].Gx, point_y=_curves['ed448'].Gy)
        self.assertRaises(ValueError, ECC.construct, curve='Ed448', **coord)
        self.assertRaises(ValueError, ECC.construct, curve='Ed448', d=2, **coordG)
        self.assertRaises(ValueError, ECC.construct, curve='Ed448', seed=b'H' * 58)