import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
class TestOFPActionSetTpDst(unittest.TestCase):
    """ Test case for ofproto_v1_0_parser.OFPActionSetTpDst
    """
    type_ = {'buf': b'\x00\n', 'val': ofproto.OFPAT_SET_TP_DST}
    len_ = {'buf': b'\x00\x08', 'val': ofproto.OFP_ACTION_TP_PORT_SIZE}
    tp = {'buf': b'\x06m', 'val': 1645}
    zfill = b'\x00' * 2
    buf = type_['buf'] + len_['buf'] + tp['buf'] + zfill
    c = OFPActionSetTpDst(tp['val'])

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.tp['val'], self.c.tp)

    def test_parser_dst(self):
        res = self.c.parser(self.buf, 0)
        self.assertEqual(self.tp['val'], res.tp)

    def test_parser_src(self):
        type_ = {'buf': b'\x00\t', 'val': ofproto.OFPAT_SET_TP_SRC}
        buf = type_['buf'] + self.len_['buf'] + self.tp['buf'] + self.zfill
        res = self.c.parser(buf, 0)
        self.assertEqual(self.tp['val'], res.tp)

    def test_parser_check_type(self):
        type_ = {'buf': b'\x00\x10', 'val': 16}
        buf = type_['buf'] + self.len_['buf'] + self.tp['buf'] + self.zfill
        self.assertRaises(AssertionError, self.c.parser, buf, 0)

    def test_parser_check_len(self):
        len_ = {'buf': b'\x00\x07', 'val': 7}
        buf = self.type_['buf'] + len_['buf'] + self.tp['buf'] + self.zfill
        self.assertRaises(AssertionError, self.c.parser, buf, 0)

    def test_serialize(self):
        buf = bytearray()
        self.c.serialize(buf, 0)
        fmt = ofproto.OFP_ACTION_TP_PORT_PACK_STR
        res = struct.unpack(fmt, bytes(buf))
        self.assertEqual(self.type_['val'], res[0])
        self.assertEqual(self.len_['val'], res[1])
        self.assertEqual(self.tp['val'], res[2])