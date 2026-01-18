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
class TestOFPActionDecMplsTtl(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPActionDecMplsTtl
    """
    type_ = ofproto.OFPAT_DEC_MPLS_TTL
    len_ = ofproto.OFP_ACTION_MPLS_TTL_SIZE
    fmt = ofproto.OFP_ACTION_HEADER_PACK_STR
    buf = pack(fmt, type_, len_)
    c = OFPActionDecMplsTtl()

    def test_parser(self):
        res = self.c.parser(self.buf, 0)
        self.assertEqual(res.len, self.len_)
        self.assertEqual(res.type, self.type_)

    def test_serialize(self):
        buf = bytearray()
        self.c.serialize(buf, 0)
        res = struct.unpack(self.fmt, bytes(buf))
        self.assertEqual(res[0], self.type_)
        self.assertEqual(res[1], self.len_)