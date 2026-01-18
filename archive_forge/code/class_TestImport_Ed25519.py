import unittest
from binascii import unhexlify
from Cryptodome.PublicKey import ECC
from Cryptodome.Signature import eddsa
from Cryptodome.Hash import SHA512, SHAKE256
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.Util.number import bytes_to_long
class TestImport_Ed25519(unittest.TestCase):

    def test_raw(self):
        Px = 24407857220263921307776619664228778204996144802740950419837658238229122415920
        Py = 56480760040633817885061096979765646085062883740629155052073094891081309750690
        encoded = b'\xa2\x05\xd6\x00\xe1 \xe1\xc0\xff\x96\xee?V\x8e\xba/\xd3\x89\x06\xd7\xc4c\xe8$\xc2d\xd7a1\xfa\xde|'
        key = eddsa.import_public_key(encoded)
        self.assertEqual(Py, key.pointQ.y)
        self.assertEqual(Px, key.pointQ.x)
        encoded = b'\x01' + b'\x00' * 31
        key = eddsa.import_public_key(encoded)
        self.assertEqual(1, key.pointQ.y)
        self.assertEqual(0, key.pointQ.x)