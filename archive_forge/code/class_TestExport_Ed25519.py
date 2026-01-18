import unittest
from binascii import unhexlify
from Cryptodome.PublicKey import ECC
from Cryptodome.Signature import eddsa
from Cryptodome.Hash import SHA512, SHAKE256
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.Util.number import bytes_to_long
class TestExport_Ed25519(unittest.TestCase):

    def test_raw(self):
        key = ECC.generate(curve='Ed25519')
        x, y = key.pointQ.xy
        raw = bytearray(key._export_eddsa())
        sign_x = raw[31] >> 7
        raw[31] &= 127
        yt = bytes_to_long(raw[::-1])
        self.assertEqual(y, yt)
        self.assertEqual(x & 1, sign_x)
        key = ECC.construct(point_x=0, point_y=1, curve='Ed25519')
        out = key._export_eddsa()
        self.assertEqual(b'\x01' + b'\x00' * 31, out)