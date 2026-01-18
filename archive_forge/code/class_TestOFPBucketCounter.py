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
class TestOFPBucketCounter(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPBucketCounter
    """
    packet_count = 6489108735192644493
    byte_count = 7334344481123449724

    def test_init(self):
        c = OFPBucketCounter(self.packet_count, self.byte_count)
        self.assertEqual(self.packet_count, c.packet_count)
        self.assertEqual(self.byte_count, c.byte_count)

    def _test_parser(self, packet_count, byte_count):
        fmt = ofproto.OFP_BUCKET_COUNTER_PACK_STR
        buf = pack(fmt, packet_count, byte_count)
        res = OFPBucketCounter.parser(buf, 0)
        self.assertEqual(packet_count, res.packet_count)
        self.assertEqual(byte_count, res.byte_count)

    def test_parser_mid(self):
        self._test_parser(self.packet_count, self.byte_count)

    def test_parser_max(self):
        packet_count = 18446744073709551615
        byte_count = 18446744073709551615
        self._test_parser(packet_count, byte_count)

    def test_parser_min(self):
        packet_count = 0
        byte_count = 0
        self._test_parser(packet_count, byte_count)