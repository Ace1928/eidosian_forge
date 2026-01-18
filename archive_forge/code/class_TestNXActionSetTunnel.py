import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
class TestNXActionSetTunnel(unittest.TestCase):
    """ Test case for ofproto_v1_0_parser.NXActionSetTunnel
    """
    type_ = {'buf': b'\xff\xff', 'val': ofproto.OFPAT_VENDOR}
    len_ = {'buf': b'\x00\x10', 'val': ofproto.NX_ACTION_SET_TUNNEL_SIZE}
    vendor = {'buf': b'\x00\x00# ', 'val': 8992}
    subtype = {'buf': b'\x00\x02', 'val': 2}
    zfill = b'\x00' * 2
    tun_id = {'buf': b'\x01o\x01\xd0', 'val': 24052176}
    buf = type_['buf'] + len_['buf'] + vendor['buf'] + subtype['buf'] + zfill + tun_id['buf']
    c = NXActionSetTunnel(tun_id['val'])

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.subtype['val'], self.c.subtype)
        self.assertEqual(self.tun_id['val'], self.c.tun_id)

    def test_parse(self):
        res = OFPActionVendor.parser(self.buf, 0)
        self.assertEqual(self.type_['val'], res.type)
        self.assertEqual(self.len_['val'], res.len)
        self.assertEqual(self.tun_id['val'], res.tun_id)

    def test_serialize(self):
        buf = bytearray()
        self.c.serialize(buf, 0)
        fmt = ofproto.NX_ACTION_SET_TUNNEL_PACK_STR
        res = struct.unpack(fmt, bytes(buf))
        self.assertEqual(self.type_['val'], res[0])
        self.assertEqual(self.len_['val'], res[1])
        self.assertEqual(self.vendor['val'], res[2])
        self.assertEqual(self.subtype['val'], res[3])
        self.assertEqual(self.tun_id['val'], res[4])