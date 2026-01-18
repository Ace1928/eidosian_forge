import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
class TestOFPActionVlanPcp(unittest.TestCase):
    """ Test case for ofproto_v1_0_parser.OFPActionVlanPcp
    """
    type_ = {'buf': b'\x00\x02', 'val': ofproto.OFPAT_SET_VLAN_PCP}
    len_ = {'buf': b'\x00\x08', 'val': ofproto.OFP_ACTION_VLAN_PCP_SIZE}
    vlan_pcp = {'buf': b'\x1c', 'val': 28}
    zfill = b'\x00' * 3
    buf = type_['buf'] + len_['buf'] + vlan_pcp['buf'] + zfill
    c = OFPActionVlanPcp(vlan_pcp['val'])

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.vlan_pcp['val'], self.c.vlan_pcp)

    def test_parser(self):
        res = self.c.parser(self.buf, 0)
        self.assertEqual(self.vlan_pcp['val'], res.vlan_pcp)

    def test_parser_check_type(self):
        type_ = {'buf': b'\x00\x01', 'val': 1}
        buf = type_['buf'] + self.len_['buf'] + self.vlan_pcp['buf'] + self.zfill
        self.assertRaises(AssertionError, self.c.parser, buf, 0)

    def test_parser_check_len(self):
        len_ = {'buf': b'\x00\x07', 'val': 7}
        buf = self.type_['buf'] + len_['buf'] + self.vlan_pcp['buf'] + self.zfill
        self.assertRaises(AssertionError, self.c.parser, buf, 0)

    def test_serialize(self):
        buf = bytearray()
        self.c.serialize(buf, 0)
        fmt = ofproto.OFP_ACTION_VLAN_PCP_PACK_STR
        res = struct.unpack(fmt, bytes(buf))
        self.assertEqual(self.type_['val'], res[0])
        self.assertEqual(self.len_['val'], res[1])
        self.assertEqual(self.vlan_pcp['val'], res[2])