import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
class TestNXActionResubmitTable(unittest.TestCase):
    """ Test case for ofproto_v1_0_parser.NXActionResubmitTable
    """
    type_ = {'buf': b'\xff\xff', 'val': ofproto.OFPAT_VENDOR}
    len_ = {'buf': b'\x00\x10', 'val': ofproto.NX_ACTION_RESUBMIT_SIZE}
    vendor = {'buf': b'\x00\x00# ', 'val': 8992}
    subtype = {'buf': b'\x00\x0e', 'val': 14}
    in_port = {'buf': b'\nL', 'val': 2636}
    table_id = {'buf': b'R', 'val': 82}
    zfill = b'\x00' * 3
    buf = type_['buf'] + len_['buf'] + vendor['buf'] + subtype['buf'] + in_port['buf'] + table_id['buf'] + zfill
    c = NXActionResubmitTable(in_port['val'], table_id['val'])

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.subtype['val'], self.c.subtype)
        self.assertEqual(self.in_port['val'], self.c.in_port)
        self.assertEqual(self.table_id['val'], self.c.table_id)

    def test_parser(self):
        res = OFPActionVendor.parser(self.buf, 0)
        self.assertEqual(self.type_['val'], res.type)
        self.assertEqual(self.len_['val'], res.len)
        self.assertEqual(self.in_port['val'], res.in_port)
        self.assertEqual(self.table_id['val'], res.table_id)

    def test_serialize(self):
        buf = bytearray()
        self.c.serialize(buf, 0)
        fmt = ofproto.NX_ACTION_RESUBMIT_PACK_STR
        res = struct.unpack(fmt, bytes(buf))
        self.assertEqual(self.type_['val'], res[0])
        self.assertEqual(self.len_['val'], res[1])
        self.assertEqual(self.vendor['val'], res[2])
        self.assertEqual(self.subtype['val'], res[3])
        self.assertEqual(self.in_port['val'], res[4])
        self.assertEqual(self.table_id['val'], res[5])