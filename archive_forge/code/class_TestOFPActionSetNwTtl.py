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
class TestOFPActionSetNwTtl(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPActionSetNwTtl
    """
    type_ = ofproto.OFPAT_SET_NW_TTL
    len_ = ofproto.OFP_ACTION_NW_TTL_SIZE
    nw_ttl = 240
    fmt = ofproto.OFP_ACTION_NW_TTL_PACK_STR

    def test_init(self):
        c = OFPActionSetNwTtl(self.nw_ttl)
        self.assertEqual(self.nw_ttl, c.nw_ttl)

    def _test_parser(self, nw_ttl):
        buf = pack(self.fmt, self.type_, self.len_, nw_ttl)
        res = OFPActionSetNwTtl.parser(buf, 0)
        self.assertEqual(res.type, self.type_)
        self.assertEqual(res.len, self.len_)
        self.assertEqual(res.nw_ttl, nw_ttl)

    def test_parser_mid(self):
        self._test_parser(self.nw_ttl)

    def test_parser_max(self):
        self._test_parser(255)

    def test_parser_min(self):
        self._test_parser(0)

    def _test_serialize(self, nw_ttl):
        c = OFPActionSetNwTtl(nw_ttl)
        buf = bytearray()
        c.serialize(buf, 0)
        res = struct.unpack(self.fmt, bytes(buf))
        self.assertEqual(res[0], self.type_)
        self.assertEqual(res[1], self.len_)
        self.assertEqual(res[2], nw_ttl)

    def test_serialize_mid(self):
        self._test_serialize(self.nw_ttl)

    def test_serialize_max(self):
        self._test_serialize(255)

    def test_serialize_min(self):
        self._test_serialize(0)