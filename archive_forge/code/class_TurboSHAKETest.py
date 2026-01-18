import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import TurboSHAKE128, TurboSHAKE256
from Cryptodome.Util.py3compat import bchr
class TurboSHAKETest(unittest.TestCase):

    def test_new_positive(self):
        xof1 = self.TurboSHAKE.new()
        xof1.update(b'90')
        xof2 = self.TurboSHAKE.new(domain=31)
        xof2.update(b'90')
        xof3 = self.TurboSHAKE.new(data=b'90')
        out1 = xof1.read(128)
        out2 = xof2.read(128)
        out3 = xof3.read(128)
        self.assertEqual(out1, out2)
        self.assertEqual(out1, out3)

    def test_new_domain(self):
        xof1 = self.TurboSHAKE.new(domain=29)
        xof2 = self.TurboSHAKE.new(domain=32)
        self.assertNotEqual(xof1.read(128), xof2.read(128))

    def test_update(self):
        pieces = [bchr(10) * 200, bchr(20) * 300]
        xof1 = self.TurboSHAKE.new()
        xof1.update(pieces[0]).update(pieces[1])
        digest1 = xof1.read(10)
        xof2 = self.TurboSHAKE.new()
        xof2.update(pieces[0] + pieces[1])
        digest2 = xof2.read(10)
        self.assertEqual(digest1, digest2)

    def test_update_negative(self):
        xof1 = self.TurboSHAKE.new()
        self.assertRaises(TypeError, xof1.update, u'string')

    def test_read(self):
        xof1 = self.TurboSHAKE.new()
        digest = xof1.read(90)
        self.assertTrue(isinstance(digest, bytes))
        self.assertEqual(len(digest), 90)

    def test_update_after_read(self):
        xof1 = self.TurboSHAKE.new()
        xof1.update(b'rrrr')
        xof1.read(90)
        self.assertRaises(TypeError, xof1.update, b'ttt')

    def test_new(self):
        xof1 = self.TurboSHAKE.new(domain=7)
        xof1.update(b'90')
        digest1 = xof1.read(100)
        xof2 = xof1.new()
        xof2.update(b'90')
        digest2 = xof2.read(100)
        self.assertEqual(digest1, digest2)
        self.assertRaises(TypeError, xof1.new, domain=7)