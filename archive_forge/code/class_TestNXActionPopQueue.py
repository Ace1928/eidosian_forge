import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
class TestNXActionPopQueue(unittest.TestCase):
    """ Test case for ofproto_v1_0_parser.NXActionPopQueue
    """
    type_ = {'buf': b'\xff\xff', 'val': ofproto.OFPAT_VENDOR}
    len_ = {'buf': b'\x00\x10', 'val': ofproto.NX_ACTION_SET_TUNNEL_SIZE}
    vendor = {'buf': b'\x00\x00# ', 'val': ofproto_common.NX_EXPERIMENTER_ID}
    subtype = {'buf': b'\x00\x05', 'val': ofproto.NXAST_POP_QUEUE}
    zfill = b'\x00' * 6
    buf = type_['buf'] + len_['buf'] + vendor['buf'] + subtype['buf'] + zfill
    c = NXActionPopQueue()

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.subtype['val'], self.c.subtype)

    def test_parser(self):
        res = OFPActionVendor.parser(self.buf, 0)
        self.assertEqual(self.type_['val'], res.type)
        self.assertEqual(self.len_['val'], res.len)
        self.assertEqual(self.subtype['val'], res.subtype)

    def test_serialize(self):
        buf = bytearray()
        self.c.serialize(buf, 0)
        fmt = ofproto.NX_ACTION_POP_QUEUE_PACK_STR
        res = struct.unpack(fmt, bytes(buf))
        self.assertEqual(self.type_['val'], res[0])
        self.assertEqual(self.len_['val'], res[1])
        self.assertEqual(self.vendor['val'], res[2])
        self.assertEqual(self.subtype['val'], res[3])