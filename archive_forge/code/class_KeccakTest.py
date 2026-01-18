import unittest
from binascii import hexlify, unhexlify
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import keccak
from Cryptodome.Util.py3compat import b, tobytes, bchr
class KeccakTest(unittest.TestCase):

    def test_new_positive(self):
        for digest_bits in (224, 256, 384, 512):
            hobj = keccak.new(digest_bits=digest_bits)
            self.assertEqual(hobj.digest_size, digest_bits // 8)
            hobj2 = hobj.new()
            self.assertEqual(hobj2.digest_size, digest_bits // 8)
        for digest_bytes in (28, 32, 48, 64):
            hobj = keccak.new(digest_bytes=digest_bytes)
            self.assertEqual(hobj.digest_size, digest_bytes)
            hobj2 = hobj.new()
            self.assertEqual(hobj2.digest_size, digest_bytes)

    def test_new_positive2(self):
        digest1 = keccak.new(data=b('\x90'), digest_bytes=64).digest()
        digest2 = keccak.new(digest_bytes=64).update(b('\x90')).digest()
        self.assertEqual(digest1, digest2)

    def test_new_negative(self):
        self.assertRaises(TypeError, keccak.new)
        h = keccak.new(digest_bits=512)
        self.assertRaises(TypeError, keccak.new, digest_bytes=64, digest_bits=512)
        self.assertRaises(ValueError, keccak.new, digest_bytes=0)
        self.assertRaises(ValueError, keccak.new, digest_bytes=1)
        self.assertRaises(ValueError, keccak.new, digest_bytes=65)
        self.assertRaises(ValueError, keccak.new, digest_bits=0)
        self.assertRaises(ValueError, keccak.new, digest_bits=1)
        self.assertRaises(ValueError, keccak.new, digest_bits=513)

    def test_update(self):
        pieces = [bchr(10) * 200, bchr(20) * 300]
        h = keccak.new(digest_bytes=64)
        h.update(pieces[0]).update(pieces[1])
        digest = h.digest()
        h = keccak.new(digest_bytes=64)
        h.update(pieces[0] + pieces[1])
        self.assertEqual(h.digest(), digest)

    def test_update_negative(self):
        h = keccak.new(digest_bytes=64)
        self.assertRaises(TypeError, h.update, u'string')

    def test_digest(self):
        h = keccak.new(digest_bytes=64)
        digest = h.digest()
        self.assertEqual(h.digest(), digest)
        self.assertTrue(isinstance(digest, type(b('digest'))))

    def test_hex_digest(self):
        mac = keccak.new(digest_bits=512)
        digest = mac.digest()
        hexdigest = mac.hexdigest()
        self.assertEqual(hexlify(digest), tobytes(hexdigest))
        self.assertEqual(mac.hexdigest(), hexdigest)
        self.assertTrue(isinstance(hexdigest, type('digest')))

    def test_update_after_digest(self):
        msg = b('rrrrttt')
        h = keccak.new(digest_bits=512, data=msg[:4])
        dig1 = h.digest()
        self.assertRaises(TypeError, h.update, msg[4:])
        dig2 = keccak.new(digest_bits=512, data=msg).digest()
        h = keccak.new(digest_bits=512, data=msg[:4], update_after_digest=True)
        self.assertEqual(h.digest(), dig1)
        h.update(msg[4:])
        self.assertEqual(h.digest(), dig2)