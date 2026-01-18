import unittest
from binascii import unhexlify, hexlify
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import TupleHash128, TupleHash256
class TupleHashTest(unittest.TestCase):

    def new(self, *args, **kwargs):
        return self.TupleHash.new(*args, **kwargs)

    def test_new_positive(self):
        h = self.new()
        for new_func in (self.TupleHash.new, h.new):
            for dbits in range(64, 1024 + 1, 8):
                hobj = new_func(digest_bits=dbits)
                self.assertEqual(hobj.digest_size * 8, dbits)
            for dbytes in range(8, 128 + 1):
                hobj = new_func(digest_bytes=dbytes)
                self.assertEqual(hobj.digest_size, dbytes)
        hobj = h.new()
        self.assertEqual(hobj.digest_size, self.default_bytes)

    def test_new_negative(self):
        h = self.new()
        for new_func in (self.TupleHash.new, h.new):
            self.assertRaises(TypeError, new_func, digest_bytes=self.minimum_bytes, digest_bits=self.minimum_bits)
            self.assertRaises(ValueError, new_func, digest_bytes=0)
            self.assertRaises(ValueError, new_func, digest_bits=self.minimum_bits + 7)
            self.assertRaises(ValueError, new_func, digest_bits=self.minimum_bits - 8)
            self.assertRaises(ValueError, new_func, digest_bits=self.minimum_bytes - 1)

    def test_default_digest_size(self):
        digest = self.new().digest()
        self.assertEqual(len(digest), self.default_bytes)

    def test_update(self):
        h = self.new()
        h.update(b'')
        h.digest()
        h = self.new()
        h.update(b'')
        h.update(b'STRING1')
        h.update(b'STRING2')
        mac1 = h.digest()
        h = self.new()
        h.update(b'STRING1')
        h.update(b'STRING2')
        mac2 = h.digest()
        self.assertNotEqual(mac1, mac2)
        h = self.new()
        h.update(b'STRING1', b'STRING2')
        self.assertEqual(mac2, h.digest())
        h = self.new()
        t = (b'STRING1', b'STRING2')
        h.update(*t)
        self.assertEqual(mac2, h.digest())

    def test_update_negative(self):
        h = self.new()
        self.assertRaises(TypeError, h.update, u'string')
        self.assertRaises(TypeError, h.update, None)
        self.assertRaises(TypeError, h.update, (b'STRING1', b'STRING2'))

    def test_digest(self):
        h = self.new()
        digest = h.digest()
        self.assertEqual(h.digest(), digest)
        self.assertTrue(isinstance(digest, type(b'digest')))

    def test_update_after_digest(self):
        msg = b'rrrrttt'
        h = self.new()
        h.update(msg)
        dig1 = h.digest()
        self.assertRaises(TypeError, h.update, dig1)

    def test_hex_digest(self):
        mac = self.new()
        digest = mac.digest()
        hexdigest = mac.hexdigest()
        self.assertEqual(hexlify(digest), tobytes(hexdigest))
        self.assertEqual(mac.hexdigest(), hexdigest)
        self.assertTrue(isinstance(hexdigest, type('digest')))

    def test_bytearray(self):
        data = b'\x00\x01\x02'
        data_ba = bytearray(data)
        h1 = self.new()
        h2 = self.new()
        h1.update(data)
        h2.update(data_ba)
        data_ba[:1] = b'\xff'
        self.assertEqual(h1.digest(), h2.digest())

    def test_memoryview(self):
        data = b'\x00\x01\x02'

        def get_mv_ro(data):
            return memoryview(data)

        def get_mv_rw(data):
            return memoryview(bytearray(data))
        for get_mv in (get_mv_ro, get_mv_rw):
            data_mv = get_mv(data)
            h1 = self.new()
            h2 = self.new()
            h1.update(data)
            h2.update(data_mv)
            if not data_mv.readonly:
                data_mv[:1] = b'\xff'
            self.assertEqual(h1.digest(), h2.digest())