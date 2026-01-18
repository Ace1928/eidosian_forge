import unittest
from binascii import unhexlify
from Cryptodome.PublicKey import ECC
from Cryptodome.Signature import eddsa
from Cryptodome.Hash import SHA512, SHAKE256
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.Util.number import bytes_to_long
class TestImport_Ed448(unittest.TestCase):

    def test_raw(self):
        Px = 60325061579351097531396300007108771711093478265463072719503110997960651731715912034033372399020294199455638771824906386451257058064817
        Py = 161721505479853217674416112915620066142532235043030707115194901038152094349490517802829490596468509257123434644399234827419409519319845
        encoded = unhexlify('259b71c19f83ef77a7abd26524cbdb3161b590a48f7d17de3ee0ba9c52beb743c09428a131d6b1b57303d90d8132c276d5ed3d5d01c0f53880')
        key = eddsa.import_public_key(encoded)
        self.assertEqual(Py, key.pointQ.y)
        self.assertEqual(Px, key.pointQ.x)
        encoded = b'\x01' + b'\x00' * 56
        key = eddsa.import_public_key(encoded)
        self.assertEqual(1, key.pointQ.y)
        self.assertEqual(0, key.pointQ.x)