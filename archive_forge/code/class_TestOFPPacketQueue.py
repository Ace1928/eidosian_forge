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
class TestOFPPacketQueue(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPPacketQueue
    """

    def test_init(self):
        queue_id = 1
        port = 2
        len_ = 3
        properties = [4, 5, 6]
        c = OFPPacketQueue(queue_id, port, properties)
        self.assertEqual(queue_id, c.queue_id)
        self.assertEqual(port, c.port)
        self.assertEqual(properties, c.properties)

    def _test_parser(self, queue_id, port, prop_cnt):
        fmt = ofproto.OFP_PACKET_QUEUE_PACK_STR
        queue_len = ofproto.OFP_PACKET_QUEUE_SIZE + ofproto.OFP_QUEUE_PROP_MIN_RATE_SIZE * prop_cnt
        buf = pack(fmt, queue_id, port, queue_len)
        for rate in range(prop_cnt):
            fmt = ofproto.OFP_QUEUE_PROP_HEADER_PACK_STR
            prop_type = ofproto.OFPQT_MIN_RATE
            prop_len = ofproto.OFP_QUEUE_PROP_MIN_RATE_SIZE
            buf += pack(fmt, prop_type, prop_len)
            fmt = ofproto.OFP_QUEUE_PROP_MIN_RATE_PACK_STR
            prop_rate = rate
            buf += pack(fmt, prop_rate)
        res = OFPPacketQueue.parser(buf, 0)
        self.assertEqual(queue_id, res.queue_id)
        self.assertEqual(port, res.port)
        self.assertEqual(queue_len, res.len)
        self.assertEqual(prop_cnt, len(res.properties))
        for rate, p in enumerate(res.properties):
            self.assertEqual(prop_type, p.property)
            self.assertEqual(prop_len, p.len)
            self.assertEqual(rate, p.rate)

    def test_parser_mid(self):
        queue_id = 1
        port = 2
        prop_cnt = 2
        self._test_parser(queue_id, port, prop_cnt)

    def test_parser_max(self):
        queue_id = 4294967295
        port = 4294967295
        prop_cnt = 4094
        self._test_parser(queue_id, port, prop_cnt)

    def test_parser_min(self):
        queue_id = 0
        port = 0
        prop_cnt = 0
        self._test_parser(queue_id, port, prop_cnt)