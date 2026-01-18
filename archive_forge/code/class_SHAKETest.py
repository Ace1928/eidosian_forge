import unittest
from binascii import hexlify, unhexlify
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import SHAKE128, SHAKE256
from Cryptodome.Util.py3compat import b, bchr, bord, tobytes
class SHAKETest(unittest.TestCase):

    def test_new_positive(self):
        xof1 = self.shake.new()
        xof2 = self.shake.new(data=b('90'))
        xof3 = self.shake.new().update(b('90'))
        self.assertNotEqual(xof1.read(10), xof2.read(10))
        xof3.read(10)
        self.assertEqual(xof2.read(10), xof3.read(10))

    def test_update(self):
        pieces = [bchr(10) * 200, bchr(20) * 300]
        h = self.shake.new()
        h.update(pieces[0]).update(pieces[1])
        digest = h.read(10)
        h = self.shake.new()
        h.update(pieces[0] + pieces[1])
        self.assertEqual(h.read(10), digest)

    def test_update_negative(self):
        h = self.shake.new()
        self.assertRaises(TypeError, h.update, u'string')

    def test_digest(self):
        h = self.shake.new()
        digest = h.read(90)
        self.assertTrue(isinstance(digest, type(b('digest'))))
        self.assertEqual(len(digest), 90)

    def test_update_after_read(self):
        mac = self.shake.new()
        mac.update(b('rrrr'))
        mac.read(90)
        self.assertRaises(TypeError, mac.update, b('ttt'))