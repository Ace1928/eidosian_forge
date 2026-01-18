import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib.packet import ipv6
class Test_fragment(unittest.TestCase):

    def setUp(self):
        self.nxt = 44
        self.offset = 50
        self.more = 1
        self.id_ = 123
        self.fragment = ipv6.fragment(self.nxt, self.offset, self.more, self.id_)
        self.off_m = self.offset << 3 | self.more
        self.form = '!BxHI'
        self.buf = struct.pack(self.form, self.nxt, self.off_m, self.id_)

    def test_init(self):
        self.assertEqual(self.nxt, self.fragment.nxt)
        self.assertEqual(self.offset, self.fragment.offset)
        self.assertEqual(self.more, self.fragment.more)
        self.assertEqual(self.id_, self.fragment.id_)

    def test_parser(self):
        _res = ipv6.fragment.parser(self.buf)
        if type(_res) is tuple:
            res = _res[0]
        else:
            res = _res
        self.assertEqual(self.nxt, res.nxt)
        self.assertEqual(self.offset, res.offset)
        self.assertEqual(self.more, res.more)
        self.assertEqual(self.id_, res.id_)

    def test_serialize(self):
        buf = self.fragment.serialize()
        res = struct.unpack_from(self.form, bytes(buf))
        self.assertEqual(self.nxt, res[0])
        self.assertEqual(self.off_m, res[1])
        self.assertEqual(self.id_, res[2])

    def test_len(self):
        self.assertEqual(8, len(self.fragment))

    def test_default_args(self):
        hdr = ipv6.fragment()
        buf = hdr.serialize()
        res = struct.unpack_from(ipv6.fragment._PACK_STR, buf)
        self.assertEqual(res[0], 6)
        self.assertEqual(res[1], 0)
        self.assertEqual(res[2], 0)