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
class TestOFPHello(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPHello
    """

    def _test_parser(self, xid):
        version = ofproto.OFP_VERSION
        msg_type = ofproto.OFPT_HELLO
        msg_len = ofproto.OFP_HEADER_SIZE
        fmt = ofproto.OFP_HEADER_PACK_STR
        buf = pack(fmt, version, msg_type, msg_len, xid)
        res = OFPHello.parser(object, version, msg_type, msg_len, xid, bytearray(buf))
        self.assertEqual(version, res.version)
        self.assertEqual(msg_type, res.msg_type)
        self.assertEqual(msg_len, res.msg_len)
        self.assertEqual(xid, res.xid)
        self.assertEqual(bytes(buf), bytes(res.buf))

    def test_parser_xid_min(self):
        xid = 0
        self._test_parser(xid)

    def test_parser_xid_mid(self):
        xid = 2183948390
        self._test_parser(xid)

    def test_parser_xid_max(self):
        xid = 4294967295
        self._test_parser(xid)

    def test_serialize(self):
        c = OFPHello(_Datapath)
        c.serialize()
        self.assertEqual(ofproto.OFP_VERSION, c.version)
        self.assertEqual(ofproto.OFPT_HELLO, c.msg_type)
        self.assertEqual(0, c.xid)