import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
class TestNXActionRegMove(unittest.TestCase):
    """ Test case for ofproto_v1_0_parser.NXActionRegMove
    """
    type_ = {'buf': b'\xff\xff', 'val': ofproto.OFPAT_VENDOR}
    len_ = {'buf': b'\x00\x18', 'val': ofproto.NX_ACTION_REG_MOVE_SIZE}
    vendor = {'buf': b'\x00\x00# ', 'val': ofproto_common.NX_EXPERIMENTER_ID}
    subtype = {'buf': b'\x00\x06', 'val': ofproto.NXAST_REG_MOVE}
    n_bits = {'buf': b'=\x98', 'val': 15768}
    src_ofs = {'buf': b'\xf3\xa3', 'val': 62371}
    dst_ofs = {'buf': b'\xdcg', 'val': 56423}
    src_field = {'buf': b'\x00\x01\x00\x04', 'val': 'reg0', 'val2': 65540}
    dst_field = {'buf': b'\x00\x01\x02\x04', 'val': 'reg1', 'val2': 66052}
    buf = type_['buf'] + len_['buf'] + vendor['buf'] + subtype['buf'] + n_bits['buf'] + src_ofs['buf'] + dst_ofs['buf'] + src_field['buf'] + dst_field['buf']
    c = NXActionRegMove(src_field['val'], dst_field['val'], n_bits['val'], src_ofs['val'], dst_ofs['val'])

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.subtype['val'], self.c.subtype)
        self.assertEqual(self.src_field['val'], self.c.src_field)
        self.assertEqual(self.dst_field['val'], self.c.dst_field)
        self.assertEqual(self.n_bits['val'], self.c.n_bits)
        self.assertEqual(self.src_field['val'], self.c.src_field)
        self.assertEqual(self.dst_field['val'], self.c.dst_field)

    def test_parser(self):
        res = OFPActionVendor.parser(self.buf, 0)
        self.assertEqual(self.type_['val'], res.type)
        self.assertEqual(self.len_['val'], res.len)
        self.assertEqual(self.subtype['val'], res.subtype)
        self.assertEqual(self.src_ofs['val'], res.src_ofs)
        self.assertEqual(self.dst_ofs['val'], res.dst_ofs)
        self.assertEqual(self.n_bits['val'], res.n_bits)
        self.assertEqual(self.src_field['val'], res.src_field)
        self.assertEqual(self.dst_field['val'], res.dst_field)

    def test_serialize(self):
        buf = bytearray()
        self.c.serialize(buf, 0)
        fmt = ofproto.NX_ACTION_REG_MOVE_PACK_STR
        res = struct.unpack(fmt, bytes(buf))
        self.assertEqual(self.type_['val'], res[0])
        self.assertEqual(self.len_['val'], res[1])
        self.assertEqual(self.vendor['val'], res[2])
        self.assertEqual(self.subtype['val'], res[3])
        self.assertEqual(self.n_bits['val'], res[4])
        self.assertEqual(self.src_ofs['val'], res[5])
        self.assertEqual(self.dst_ofs['val'], res[6])
        self.assertEqual(self.src_field['val2'], res[7])
        self.assertEqual(self.dst_field['val2'], res[8])