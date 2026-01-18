import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib.packet import ipv6
class Test_routing_type3(unittest.TestCase):

    def setUp(self):
        self.nxt = 0
        self.size = 6
        self.type_ = 3
        self.seg = 0
        self.cmpi = 0
        self.cmpe = 0
        self.adrs = ['2001:db8:dead::1', '2001:db8:dead::2', '2001:db8:dead::3']
        self.pad = (8 - ((len(self.adrs) - 1) * (16 - self.cmpi) + (16 - self.cmpe) % 8)) % 8
        self.routing = ipv6.routing_type3(self.nxt, self.size, self.type_, self.seg, self.cmpi, self.cmpe, self.adrs)
        self.form = '!BBBBBB2x16s16s16s'
        self.buf = struct.pack(self.form, self.nxt, self.size, self.type_, self.seg, self.cmpi << 4 | self.cmpe, self.pad << 4, addrconv.ipv6.text_to_bin(self.adrs[0]), addrconv.ipv6.text_to_bin(self.adrs[1]), addrconv.ipv6.text_to_bin(self.adrs[2]))

    def test_init(self):
        self.assertEqual(self.nxt, self.routing.nxt)
        self.assertEqual(self.size, self.routing.size)
        self.assertEqual(self.type_, self.routing.type_)
        self.assertEqual(self.seg, self.routing.seg)
        self.assertEqual(self.cmpi, self.routing.cmpi)
        self.assertEqual(self.cmpe, self.routing.cmpe)
        self.assertEqual(self.pad, self.routing._pad)
        self.assertEqual(self.adrs[0], self.routing.adrs[0])
        self.assertEqual(self.adrs[1], self.routing.adrs[1])
        self.assertEqual(self.adrs[2], self.routing.adrs[2])

    def test_parser(self):
        _res = ipv6.routing.parser(self.buf)
        if type(_res) is tuple:
            res = _res[0]
        else:
            res = _res
        self.assertEqual(self.nxt, res.nxt)
        self.assertEqual(self.size, res.size)
        self.assertEqual(self.type_, res.type_)
        self.assertEqual(self.seg, res.seg)
        self.assertEqual(self.cmpi, res.cmpi)
        self.assertEqual(self.cmpe, res.cmpe)
        self.assertEqual(self.pad, res._pad)
        self.assertEqual(self.adrs[0], res.adrs[0])
        self.assertEqual(self.adrs[1], res.adrs[1])
        self.assertEqual(self.adrs[2], res.adrs[2])

    def test_serialize(self):
        buf = self.routing.serialize()
        res = struct.unpack_from(self.form, bytes(buf))
        self.assertEqual(self.nxt, res[0])
        self.assertEqual(self.size, res[1])
        self.assertEqual(self.type_, res[2])
        self.assertEqual(self.seg, res[3])
        self.assertEqual(self.cmpi, res[4] >> 4)
        self.assertEqual(self.cmpe, res[4] & 15)
        self.assertEqual(self.pad, res[5])
        self.assertEqual(addrconv.ipv6.text_to_bin(self.adrs[0]), res[6])
        self.assertEqual(addrconv.ipv6.text_to_bin(self.adrs[1]), res[7])
        self.assertEqual(addrconv.ipv6.text_to_bin(self.adrs[2]), res[8])

    def test_parser_with_adrs_zero(self):
        nxt = 0
        size = 0
        type_ = 3
        seg = 0
        cmpi = 0
        cmpe = 0
        adrs = []
        pad = (8 - ((len(adrs) - 1) * (16 - cmpi) + (16 - cmpe) % 8)) % 8
        form = '!BBBBBB2x'
        buf = struct.pack(form, nxt, size, type_, seg, cmpi << 4 | cmpe, pad << 4)
        _res = ipv6.routing.parser(buf)
        if type(_res) is tuple:
            res = _res[0]
        else:
            res = _res
        self.assertEqual(nxt, res.nxt)
        self.assertEqual(size, res.size)
        self.assertEqual(type_, res.type_)
        self.assertEqual(seg, res.seg)
        self.assertEqual(cmpi, res.cmpi)
        self.assertEqual(cmpe, res.cmpe)
        self.assertEqual(pad, res._pad)

    def test_serialize_with_adrs_zero(self):
        nxt = 0
        size = 0
        type_ = 3
        seg = 0
        cmpi = 0
        cmpe = 0
        adrs = []
        pad = (8 - ((len(adrs) - 1) * (16 - cmpi) + (16 - cmpe) % 8)) % 8
        routing = ipv6.routing_type3(nxt, size, type_, seg, cmpi, cmpe, pad)
        buf = routing.serialize()
        form = '!BBBBBB2x'
        res = struct.unpack_from(form, bytes(buf))
        self.assertEqual(nxt, res[0])
        self.assertEqual(size, res[1])
        self.assertEqual(type_, res[2])
        self.assertEqual(seg, res[3])
        self.assertEqual(cmpi, res[4] >> 4)
        self.assertEqual(cmpe, res[4] & 15)
        self.assertEqual(pad, res[5])

    def test_parser_with_compression(self):
        pass
        nxt = 0
        size = 3
        type_ = 3
        seg = 0
        cmpi = 8
        cmpe = 12
        adrs = ['2001:0db8:dead:0123:4567:89ab:cdef:0001', '2001:0db8:dead:0123:4567:89ab:cdef:0002', '2001:0db8:dead:0123:4567:89ab:cdef:0003']
        pad = (8 - ((len(adrs) - 1) * (16 - cmpi) + (16 - cmpe) % 8)) % 8
        form = '!BBBBBB2x%ds%ds%ds' % (16 - cmpi, 16 - cmpi, 16 - cmpe)
        slice_i = slice(cmpi, 16)
        slice_e = slice(cmpe, 16)
        buf = struct.pack(form, nxt, size, type_, seg, cmpi << 4 | cmpe, pad << 4, addrconv.ipv6.text_to_bin(adrs[0])[slice_i], addrconv.ipv6.text_to_bin(adrs[1])[slice_i], addrconv.ipv6.text_to_bin(adrs[2])[slice_e])
        _res = ipv6.routing.parser(buf)
        if type(_res) is tuple:
            res = _res[0]
        else:
            res = _res
        self.assertEqual(nxt, res.nxt)
        self.assertEqual(size, res.size)
        self.assertEqual(type_, res.type_)
        self.assertEqual(seg, res.seg)
        self.assertEqual(cmpi, res.cmpi)
        self.assertEqual(cmpe, res.cmpe)
        self.assertEqual(pad, res._pad)
        self.assertEqual('::4567:89ab:cdef:1', res.adrs[0])
        self.assertEqual('::4567:89ab:cdef:2', res.adrs[1])
        self.assertEqual('::205.239.0.3', res.adrs[2])

    def test_serialize_with_compression(self):
        nxt = 0
        size = 3
        type_ = 3
        seg = 0
        cmpi = 8
        cmpe = 8
        adrs = ['2001:db8:dead::1', '2001:db8:dead::2', '2001:db8:dead::3']
        pad = (8 - ((len(adrs) - 1) * (16 - cmpi) + (16 - cmpe) % 8)) % 8
        slice_i = slice(cmpi, 16)
        slice_e = slice(cmpe, 16)
        routing = ipv6.routing_type3(nxt, size, type_, seg, cmpi, cmpe, adrs)
        buf = routing.serialize()
        form = '!BBBBBB2x8s8s8s'
        res = struct.unpack_from(form, bytes(buf))
        self.assertEqual(nxt, res[0])
        self.assertEqual(size, res[1])
        self.assertEqual(type_, res[2])
        self.assertEqual(seg, res[3])
        self.assertEqual(cmpi, res[4] >> 4)
        self.assertEqual(cmpe, res[4] & 15)
        self.assertEqual(pad, res[5])
        self.assertEqual(addrconv.ipv6.text_to_bin(adrs[0])[slice_i], res[6])
        self.assertEqual(addrconv.ipv6.text_to_bin(adrs[1])[slice_i], res[7])
        self.assertEqual(addrconv.ipv6.text_to_bin(adrs[2])[slice_e], res[8])

    def test_len(self):
        self.assertEqual((6 + 1) * 8, len(self.routing))

    def test_default_args(self):
        hdr = ipv6.routing_type3()
        buf = hdr.serialize()
        LOG.info(repr(buf))
        res = struct.unpack_from(ipv6.routing_type3._PACK_STR, bytes(buf))
        LOG.info(res)
        self.assertEqual(res[0], 6)
        self.assertEqual(res[1], 0)
        self.assertEqual(res[2], 3)
        self.assertEqual(res[3], 0)
        self.assertEqual(res[4], 0 << 4 | 0)
        self.assertEqual(res[5], 0)