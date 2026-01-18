import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
class TestOFPActionSetDlDst(unittest.TestCase):
    """ Test case for ofproto_v1_0_parser.OFPActionSetDlDst
    """
    type_ = {'buf': b'\x00\x05', 'val': ofproto.OFPAT_SET_DL_DST}
    len_ = {'buf': b'\x00\x10', 'val': ofproto.OFP_ACTION_DL_ADDR_SIZE}
    dl_addr = b'7H8\x9a\xf4('
    zfill = b'\x00' * 6
    buf = type_['buf'] + len_['buf'] + dl_addr + zfill
    c = OFPActionSetDlDst(dl_addr)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.dl_addr, self.c.dl_addr)

    def test_parser_type_dst(self):
        res = self.c.parser(self.buf, 0)
        self.assertEqual(self.dl_addr, res.dl_addr)

    def test_parser_type_src(self):
        type_ = {'buf': b'\x00\x04', 'val': ofproto.OFPAT_SET_DL_SRC}
        buf = type_['buf'] + self.len_['buf'] + self.dl_addr + self.zfill
        res = self.c.parser(buf, 0)
        self.assertEqual(self.dl_addr, res.dl_addr)

    def test_parser_check_type(self):
        type_ = {'buf': b'\x00\x06', 'val': 6}
        buf = type_['buf'] + self.len_['buf'] + self.dl_addr + self.zfill
        self.assertRaises(AssertionError, self.c.parser, buf, 0)

    def test_parser_check_len(self):
        len_ = {'buf': b'\x00\x07', 'val': 7}
        buf = self.type_['buf'] + len_['buf'] + self.dl_addr + self.zfill
        self.assertRaises(AssertionError, self.c.parser, buf, 0)

    def test_serialize(self):
        buf = bytearray()
        self.c.serialize(buf, 0)
        fmt = ofproto.OFP_ACTION_DL_ADDR_PACK_STR
        res = struct.unpack(fmt, bytes(buf))
        self.assertEqual(self.type_['val'], res[0])
        self.assertEqual(self.len_['val'], res[1])
        self.assertEqual(self.dl_addr, res[2])