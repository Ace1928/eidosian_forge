import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib.packet import ipv6
class Test_auth(unittest.TestCase):

    def setUp(self):
        self.nxt = 0
        self.size = 4
        self.spi = 256
        self.seq = 1
        self.data = b'!\xd3\xa9\\_\xfdM\x18F"\xb9\xf8'
        self.auth = ipv6.auth(self.nxt, self.size, self.spi, self.seq, self.data)
        self.form = '!BB2xII12s'
        self.buf = struct.pack(self.form, self.nxt, self.size, self.spi, self.seq, self.data)

    def test_init(self):
        self.assertEqual(self.nxt, self.auth.nxt)
        self.assertEqual(self.size, self.auth.size)
        self.assertEqual(self.spi, self.auth.spi)
        self.assertEqual(self.seq, self.auth.seq)
        self.assertEqual(self.data, self.auth.data)

    def test_parser(self):
        _res = ipv6.auth.parser(self.buf)
        if type(_res) is tuple:
            res = _res[0]
        else:
            res = _res
        self.assertEqual(self.nxt, res.nxt)
        self.assertEqual(self.size, res.size)
        self.assertEqual(self.spi, res.spi)
        self.assertEqual(self.seq, res.seq)
        self.assertEqual(self.data, res.data)

    def test_serialize(self):
        buf = self.auth.serialize()
        res = struct.unpack_from(self.form, bytes(buf))
        self.assertEqual(self.nxt, res[0])
        self.assertEqual(self.size, res[1])
        self.assertEqual(self.spi, res[2])
        self.assertEqual(self.seq, res[3])
        self.assertEqual(self.data, res[4])

    def test_len(self):
        self.assertEqual((4 + 2) * 4, len(self.auth))

    def test_len_re(self):
        size = 5
        auth = ipv6.auth(0, size, 256, 1, b'!\xd3\xa9\\_\xfdM\x18F"\xb9\xf8\xf8\xf8\xf8\xf8')
        self.assertEqual((size + 2) * 4, len(auth))

    def test_default_args(self):
        hdr = ipv6.auth()
        buf = hdr.serialize()
        LOG.info(repr(buf))
        res = struct.unpack_from(ipv6.auth._PACK_STR, bytes(buf))
        LOG.info(res)
        self.assertEqual(res[0], 6)
        self.assertEqual(res[1], 2)
        self.assertEqual(res[2], 0)
        self.assertEqual(res[3], 0)
        self.assertEqual(buf[ipv6.auth._MIN_LEN:], b'\x00\x00\x00\x00')