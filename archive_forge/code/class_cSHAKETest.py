import unittest
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import cSHAKE128, cSHAKE256, SHAKE128, SHAKE256
from Cryptodome.Util.py3compat import b, bchr, tobytes
class cSHAKETest(unittest.TestCase):

    def test_left_encode(self):
        from Cryptodome.Hash.cSHAKE128 import _left_encode
        self.assertEqual(_left_encode(0), b'\x01\x00')
        self.assertEqual(_left_encode(1), b'\x01\x01')
        self.assertEqual(_left_encode(256), b'\x02\x01\x00')

    def test_bytepad(self):
        from Cryptodome.Hash.cSHAKE128 import _bytepad
        self.assertEqual(_bytepad(b'', 4), b'\x01\x04\x00\x00')
        self.assertEqual(_bytepad(b'A', 4), b'\x01\x04A\x00')
        self.assertEqual(_bytepad(b'AA', 4), b'\x01\x04AA')
        self.assertEqual(_bytepad(b'AAA', 4), b'\x01\x04AAA\x00\x00\x00')
        self.assertEqual(_bytepad(b'AAAA', 4), b'\x01\x04AAAA\x00\x00')
        self.assertEqual(_bytepad(b'AAAAA', 4), b'\x01\x04AAAAA\x00')
        self.assertEqual(_bytepad(b'AAAAAA', 4), b'\x01\x04AAAAAA')
        self.assertEqual(_bytepad(b'AAAAAAA', 4), b'\x01\x04AAAAAAA\x00\x00\x00')

    def test_new_positive(self):
        xof1 = self.cshake.new()
        xof2 = self.cshake.new(data=b('90'))
        xof3 = self.cshake.new().update(b('90'))
        self.assertNotEqual(xof1.read(10), xof2.read(10))
        xof3.read(10)
        self.assertEqual(xof2.read(10), xof3.read(10))
        xof1 = self.cshake.new()
        ref = xof1.read(10)
        xof2 = self.cshake.new(custom=b(''))
        xof3 = self.cshake.new(custom=b('foo'))
        self.assertEqual(ref, xof2.read(10))
        self.assertNotEqual(ref, xof3.read(10))
        xof1 = self.cshake.new(custom=b('foo'))
        xof2 = self.cshake.new(custom=b('foo'), data=b('90'))
        xof3 = self.cshake.new(custom=b('foo')).update(b('90'))
        self.assertNotEqual(xof1.read(10), xof2.read(10))
        xof3.read(10)
        self.assertEqual(xof2.read(10), xof3.read(10))

    def test_update(self):
        pieces = [bchr(10) * 200, bchr(20) * 300]
        h = self.cshake.new()
        h.update(pieces[0]).update(pieces[1])
        digest = h.read(10)
        h = self.cshake.new()
        h.update(pieces[0] + pieces[1])
        self.assertEqual(h.read(10), digest)

    def test_update_negative(self):
        h = self.cshake.new()
        self.assertRaises(TypeError, h.update, u'string')

    def test_digest(self):
        h = self.cshake.new()
        digest = h.read(90)
        self.assertTrue(isinstance(digest, type(b('digest'))))
        self.assertEqual(len(digest), 90)

    def test_update_after_read(self):
        mac = self.cshake.new()
        mac.update(b('rrrr'))
        mac.read(90)
        self.assertRaises(TypeError, mac.update, b('ttt'))

    def test_shake(self):
        for digest_len in range(64):
            xof1 = self.cshake.new(b'TEST')
            xof2 = self.shake.new(b'TEST')
            self.assertEqual(xof1.read(digest_len), xof2.read(digest_len))