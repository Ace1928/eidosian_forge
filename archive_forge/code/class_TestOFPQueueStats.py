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
class TestOFPQueueStats(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPQueueStats
    """

    def test_init(self):
        port_no = 41186
        queue_id = 6606
        tx_bytes = 8638420181865882538
        tx_packets = 2856480458895760962
        tx_errors = 6283093430376743019
        res = OFPQueueStats(port_no, queue_id, tx_bytes, tx_packets, tx_errors)
        self.assertEqual(port_no, res.port_no)
        self.assertEqual(queue_id, res.queue_id)
        self.assertEqual(tx_bytes, res.tx_bytes)
        self.assertEqual(tx_packets, res.tx_packets)
        self.assertEqual(tx_errors, res.tx_errors)

    def _test_parser(self, port_no, queue_id, tx_bytes, tx_packets, tx_errors):
        fmt = ofproto.OFP_QUEUE_STATS_PACK_STR
        buf = pack(fmt, port_no, queue_id, tx_bytes, tx_packets, tx_errors)
        res = OFPQueueStats.parser(buf, 0)
        self.assertEqual(port_no, res.port_no)
        self.assertEqual(queue_id, res.queue_id)
        self.assertEqual(tx_bytes, res.tx_bytes)
        self.assertEqual(tx_packets, res.tx_packets)
        self.assertEqual(tx_errors, res.tx_errors)

    def test_parser_mid(self):
        port_no = 41186
        queue_id = 6606
        tx_bytes = 8638420181865882538
        tx_packets = 2856480458895760962
        tx_errors = 6283093430376743019
        self._test_parser(port_no, queue_id, tx_bytes, tx_packets, tx_errors)

    def test_parser_max(self):
        port_no = 4294967295
        queue_id = 4294967295
        tx_bytes = 18446744073709551615
        tx_packets = 18446744073709551615
        tx_errors = 18446744073709551615
        self._test_parser(port_no, queue_id, tx_bytes, tx_packets, tx_errors)

    def test_parser_min(self):
        port_no = 0
        queue_id = 0
        tx_bytes = 0
        tx_packets = 0
        tx_errors = 0
        self._test_parser(port_no, queue_id, tx_bytes, tx_packets, tx_errors)

    def _test_parser_p(self, port_no):
        queue_id = 6606
        tx_bytes = 8638420181865882538
        tx_packets = 2856480458895760962
        tx_errors = 6283093430376743019
        self._test_parser(port_no, queue_id, tx_bytes, tx_packets, tx_errors)

    def test_parser_p1(self):
        self._test_parser_p(ofproto.OFPP_MAX)

    def test_parser_p2(self):
        self._test_parser_p(ofproto.OFPP_IN_PORT)

    def test_parser_p3(self):
        self._test_parser_p(ofproto.OFPP_TABLE)

    def test_parser_p4(self):
        self._test_parser_p(ofproto.OFPP_NORMAL)

    def test_parser_p5(self):
        self._test_parser_p(ofproto.OFPP_FLOOD)

    def test_parser_p6(self):
        self._test_parser_p(ofproto.OFPP_ALL)

    def test_parser_p7(self):
        self._test_parser_p(ofproto.OFPP_CONTROLLER)

    def test_parser_p8(self):
        self._test_parser_p(ofproto.OFPP_LOCAL)