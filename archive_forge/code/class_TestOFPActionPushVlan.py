import unittest
import logging
import socket
from struct import *
from os_ken.ofproto.ofproto_v1_2_parser import *
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_protocol
from os_ken.ofproto import ether
from os_ken.ofproto.ofproto_parser import MsgBase
from os_ken import utils
from os_ken.lib import addrconv
from os_ken.lib import pack_utils
class TestOFPActionPushVlan(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPActionPushVlan
    """
    type_ = ofproto.OFPAT_PUSH_VLAN
    len_ = ofproto.OFP_ACTION_PUSH_SIZE
    fmt = ofproto.OFP_ACTION_PUSH_PACK_STR

    def test_init(self):
        ethertype = 33024
        c = OFPActionPushVlan(ethertype)
        self.assertEqual(ethertype, c.ethertype)

    def _test_parser(self, ethertype):
        buf = pack(self.fmt, self.type_, self.len_, ethertype)
        res = OFPActionPushVlan.parser(buf, 0)
        self.assertEqual(res.type, self.type_)
        self.assertEqual(res.len, self.len_)
        self.assertEqual(res.ethertype, ethertype)

    def test_parser_mid(self):
        self._test_parser(33024)

    def test_parser_max(self):
        self._test_parser(65535)

    def test_parser_min(self):
        self._test_parser(0)

    def _test_serialize(self, ethertype):
        c = OFPActionPushVlan(ethertype)
        buf = bytearray()
        c.serialize(buf, 0)
        res = struct.unpack(self.fmt, bytes(buf))
        self.assertEqual(res[0], self.type_)
        self.assertEqual(res[1], self.len_)
        self.assertEqual(res[2], ethertype)

    def test_serialize_mid(self):
        self._test_serialize(33024)

    def test_serialize_max(self):
        self._test_serialize(65535)

    def test_serialize_min(self):
        self._test_serialize(0)