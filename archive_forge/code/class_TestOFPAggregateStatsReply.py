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
class TestOFPAggregateStatsReply(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPAggregateStatsReply
    """
    packet_count = 5142202600015232219
    byte_count = 2659740543924820419
    flow_count = 1344694860

    def test_init(self):
        c = OFPAggregateStatsReply(self.packet_count, self.byte_count, self.flow_count)
        self.assertEqual(c.packet_count, self.packet_count)
        self.assertEqual(c.byte_count, self.byte_count)
        self.assertEqual(c.flow_count, self.flow_count)

    def _test_parser(self, packet_count, byte_count, flow_count):
        fmt = ofproto.OFP_AGGREGATE_STATS_REPLY_PACK_STR
        buf = pack(fmt, packet_count, byte_count, flow_count)
        res = OFPAggregateStatsReply.parser(buf, 0)
        self.assertEqual(packet_count, res.packet_count)
        self.assertEqual(byte_count, res.byte_count)
        self.assertEqual(flow_count, res.flow_count)

    def test_parser_mid(self):
        self._test_parser(self.packet_count, self.byte_count, self.flow_count)

    def test_parser_max(self):
        packet_count = 18446744073709551615
        byte_count = 18446744073709551615
        flow_count = 4294967295
        self._test_parser(packet_count, byte_count, flow_count)

    def test_parser_min(self):
        packet_count = 0
        byte_count = 0
        flow_count = 0
        self._test_parser(packet_count, byte_count, flow_count)