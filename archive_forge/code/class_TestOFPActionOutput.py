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
class TestOFPActionOutput(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPActionOutput
    """
    type_ = ofproto.OFPAT_OUTPUT
    len_ = ofproto.OFP_ACTION_OUTPUT_SIZE

    def test_init(self):
        port = 6606
        max_len = 1500
        fmt = ofproto.OFP_ACTION_OUTPUT_PACK_STR
        c = OFPActionOutput(port, max_len)
        self.assertEqual(port, c.port)
        self.assertEqual(max_len, c.max_len)

    def _test_parser(self, port, max_len):
        fmt = ofproto.OFP_ACTION_OUTPUT_PACK_STR
        buf = pack(fmt, self.type_, self.len_, port, max_len)
        c = OFPActionOutput(port, max_len)
        res = c.parser(buf, 0)
        self.assertEqual(res.len, self.len_)
        self.assertEqual(res.type, self.type_)
        self.assertEqual(res.port, port)
        self.assertEqual(res.max_len, max_len)

    def test_parser_mid(self):
        port = 6606
        max_len = 16
        self._test_parser(port, max_len)

    def test_parser_max(self):
        port = 4294967295
        max_len = 65535
        self._test_parser(port, max_len)

    def test_parser_min(self):
        port = 0
        max_len = 0
        self._test_parser(port, max_len)

    def test_parser_p1(self):
        port = 6606
        max_len = 65509
        self._test_parser(port, max_len)

    def _test_serialize(self, port, max_len):
        c = OFPActionOutput(port, max_len)
        buf = bytearray()
        c.serialize(buf, 0)
        fmt = ofproto.OFP_ACTION_OUTPUT_PACK_STR
        res = struct.unpack(fmt, bytes(buf))
        self.assertEqual(res[0], self.type_)
        self.assertEqual(res[1], self.len_)
        self.assertEqual(res[2], port)
        self.assertEqual(res[3], max_len)

    def test_serialize_mid(self):
        port = 6606
        max_len = 16
        self._test_serialize(port, max_len)

    def test_serialize_max(self):
        port = 4294967295
        max_len = 65535
        self._test_serialize(port, max_len)

    def test_serialize_min(self):
        port = 0
        max_len = 0
        self._test_serialize(port, max_len)

    def test_serialize_p1(self):
        port = 6606
        max_len = 65509
        self._test_serialize(port, max_len)