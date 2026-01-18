import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib.packet import ipv6
class Test_dst_opts(unittest.TestCase):

    def setUp(self):
        self.nxt = 60
        self.size = 8
        self.data = [ipv6.option(5, 2, b'\x00\x00'), ipv6.option(1, 0, None), ipv6.option(194, 4, b'\x00\x01\x00\x00'), ipv6.option(1, 0, None)]
        self.dst = ipv6.dst_opts(self.nxt, self.size, self.data)
        self.form = '!BB'
        self.buf = struct.pack(self.form, self.nxt, self.size) + self.data[0].serialize() + self.data[1].serialize() + self.data[2].serialize() + self.data[3].serialize()

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.nxt, self.dst.nxt)
        self.assertEqual(self.size, self.dst.size)
        self.assertEqual(self.data, self.dst.data)

    def test_invalid_size(self):
        self.assertRaises(Exception, ipv6.dst_opts, self.nxt, 1, self.data)

    def test_parser(self):
        _res = ipv6.dst_opts.parser(self.buf)
        if type(_res) is tuple:
            res = _res[0]
        else:
            res = _res
        self.assertEqual(self.nxt, res.nxt)
        self.assertEqual(self.size, res.size)
        self.assertEqual(str(self.data), str(res.data))

    def test_serialize(self):
        buf = self.dst.serialize()
        res = struct.unpack_from(self.form, bytes(buf))
        self.assertEqual(self.nxt, res[0])
        self.assertEqual(self.size, res[1])
        offset = struct.calcsize(self.form)
        opt1 = ipv6.option.parser(bytes(buf[offset:]))
        offset += len(opt1)
        opt2 = ipv6.option.parser(bytes(buf[offset:]))
        offset += len(opt2)
        opt3 = ipv6.option.parser(bytes(buf[offset:]))
        offset += len(opt3)
        opt4 = ipv6.option.parser(bytes(buf[offset:]))
        self.assertEqual(5, opt1.type_)
        self.assertEqual(2, opt1.len_)
        self.assertEqual(b'\x00\x00', opt1.data)
        self.assertEqual(1, opt2.type_)
        self.assertEqual(0, opt2.len_)
        self.assertEqual(None, opt2.data)
        self.assertEqual(194, opt3.type_)
        self.assertEqual(4, opt3.len_)
        self.assertEqual(b'\x00\x01\x00\x00', opt3.data)
        self.assertEqual(1, opt4.type_)
        self.assertEqual(0, opt4.len_)
        self.assertEqual(None, opt4.data)

    def test_len(self):
        self.assertEqual(16, len(self.dst))

    def test_default_args(self):
        hdr = ipv6.dst_opts()
        buf = hdr.serialize()
        res = struct.unpack('!BB', bytes(buf[:2]))
        self.assertEqual(res[0], 6)
        self.assertEqual(res[1], 0)
        opt = ipv6.option(type_=1, len_=4, data=b'\x00\x00\x00\x00')
        self.assertEqual(bytes(buf[2:]), opt.serialize())