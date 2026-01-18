import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
class TestOFPActionSetNwSrc(unittest.TestCase):
    """ Test case for ofproto_v1_0_parser.OFPActionSetNwSrc
    """
    type_ = {'buf': b'\x00\x06', 'val': ofproto.OFPAT_SET_NW_SRC}
    len_ = {'buf': b'\x00\x08', 'val': ofproto.OFP_ACTION_NW_ADDR_SIZE}
    nw_addr = {'buf': b'\xc0\xa8z\n', 'val': 3232266762}
    buf = type_['buf'] + len_['buf'] + nw_addr['buf']
    c = OFPActionSetNwSrc(nw_addr['val'])

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.nw_addr['val'], self.c.nw_addr)

    def test_parser_src(self):
        res = self.c.parser(self.buf, 0)
        self.assertEqual(self.nw_addr['val'], res.nw_addr)

    def test_parser_dst(self):
        type_ = {'buf': b'\x00\x07', 'val': ofproto.OFPAT_SET_NW_DST}
        buf = type_['buf'] + self.len_['buf'] + self.nw_addr['buf']
        res = self.c.parser(buf, 0)
        self.assertEqual(self.nw_addr['val'], res.nw_addr)

    def test_parser_check_type(self):
        type_ = {'buf': b'\x00\x05', 'val': 5}
        buf = type_['buf'] + self.len_['buf'] + self.nw_addr['buf']
        self.assertRaises(AssertionError, self.c.parser, buf, 0)

    def test_parser_check_len(self):
        len_ = {'buf': b'\x00\x10', 'val': 16}
        buf = self.type_['buf'] + len_['buf'] + self.nw_addr['buf']
        self.assertRaises(AssertionError, self.c.parser, buf, 0)

    def test_serialize(self):
        buf = bytearray()
        self.c.serialize(buf, 0)
        fmt = ofproto.OFP_ACTION_NW_ADDR_PACK_STR
        res = struct.unpack(fmt, bytes(buf))
        self.assertEqual(self.type_['val'], res[0])
        self.assertEqual(self.len_['val'], res[1])
        self.assertEqual(self.nw_addr['val'], res[2])