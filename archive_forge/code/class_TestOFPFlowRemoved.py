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
class TestOFPFlowRemoved(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPFlowRemoved
    """

    def _test_parser(self, xid, cookie, priority, reason, table_id, duration_sec, duration_nsec, idle_timeout, hard_timeout, packet_count, byte_count):
        version = ofproto.OFP_VERSION
        msg_type = ofproto.OFPT_FLOW_REMOVED
        msg_len = ofproto.OFP_FLOW_REMOVED_SIZE
        fmt = ofproto.OFP_HEADER_PACK_STR
        buf = pack(fmt, version, msg_type, msg_len, xid)
        fmt = ofproto.OFP_FLOW_REMOVED_PACK_STR0
        buf += pack(fmt, cookie, priority, reason, table_id, duration_sec, duration_nsec, idle_timeout, hard_timeout, packet_count, byte_count)
        match = OFPMatch()
        buf_match = bytearray()
        match.serialize(buf_match, 0)
        buf += bytes(buf_match)
        res = OFPFlowRemoved.parser(object, version, msg_type, msg_len, xid, buf)
        self.assertEqual(version, res.version)
        self.assertEqual(msg_type, res.msg_type)
        self.assertEqual(msg_len, res.msg_len)
        self.assertEqual(xid, res.xid)
        self.assertEqual(cookie, res.cookie)
        self.assertEqual(priority, res.priority)
        self.assertEqual(reason, res.reason)
        self.assertEqual(table_id, res.table_id)
        self.assertEqual(duration_sec, res.duration_sec)
        self.assertEqual(duration_nsec, res.duration_nsec)
        self.assertEqual(idle_timeout, res.idle_timeout)
        self.assertEqual(hard_timeout, res.hard_timeout)
        self.assertEqual(packet_count, res.packet_count)
        self.assertEqual(byte_count, res.byte_count)
        self.assertTrue(hasattr(res, 'match'))
        self.assertEqual(ofproto.OFPMT_OXM, res.match.type)

    def test_parser_mid(self):
        xid = 3423224276
        cookie = 178378173441633860
        priority = 718
        reason = 128
        table_id = 169
        duration_sec = 2250548154
        duration_nsec = 2492776995
        idle_timeout = 60284
        hard_timeout = 60285
        packet_count = 6489108735192644493
        byte_count = 7334344481123449724
        self._test_parser(xid, cookie, priority, reason, table_id, duration_sec, duration_nsec, idle_timeout, hard_timeout, packet_count, byte_count)

    def test_parser_max(self):
        xid = 4294967295
        cookie = 18446744073709551615
        priority = 65535
        reason = 255
        table_id = 255
        duration_sec = 4294967295
        duration_nsec = 4294967295
        idle_timeout = 65535
        hard_timeout = 65535
        packet_count = 18446744073709551615
        byte_count = 18446744073709551615
        self._test_parser(xid, cookie, priority, reason, table_id, duration_sec, duration_nsec, idle_timeout, hard_timeout, packet_count, byte_count)

    def test_parser_min(self):
        xid = 0
        cookie = 0
        priority = 0
        reason = ofproto.OFPRR_IDLE_TIMEOUT
        table_id = 0
        duration_sec = 0
        duration_nsec = 0
        idle_timeout = 0
        hard_timeout = 0
        packet_count = 0
        byte_count = 0
        self._test_parser(xid, cookie, priority, reason, table_id, duration_sec, duration_nsec, idle_timeout, hard_timeout, packet_count, byte_count)

    def test_parser_p1(self):
        xid = 3423224276
        cookie = 178378173441633860
        priority = 718
        reason = ofproto.OFPRR_HARD_TIMEOUT
        table_id = 169
        duration_sec = 2250548154
        duration_nsec = 2492776995
        idle_timeout = 60284
        hard_timeout = 60285
        packet_count = 6489108735192644493
        byte_count = 7334344481123449724
        self._test_parser(xid, cookie, priority, reason, table_id, duration_sec, duration_nsec, idle_timeout, hard_timeout, packet_count, byte_count)

    def test_parser_p2(self):
        xid = 3423224276
        cookie = 178378173441633860
        priority = 718
        reason = ofproto.OFPRR_DELETE
        table_id = 169
        duration_sec = 2250548154
        duration_nsec = 2492776995
        idle_timeout = 60284
        hard_timeout = 60285
        packet_count = 6489108735192644493
        byte_count = 7334344481123449724
        self._test_parser(xid, cookie, priority, reason, table_id, duration_sec, duration_nsec, idle_timeout, hard_timeout, packet_count, byte_count)

    def test_parser_p3(self):
        xid = 3423224276
        cookie = 178378173441633860
        priority = 718
        reason = ofproto.OFPRR_GROUP_DELETE
        table_id = 169
        duration_sec = 2250548154
        duration_nsec = 2492776995
        idle_timeout = 60284
        hard_timeout = 60285
        packet_count = 6489108735192644493
        byte_count = 7334344481123449724
        self._test_parser(xid, cookie, priority, reason, table_id, duration_sec, duration_nsec, idle_timeout, hard_timeout, packet_count, byte_count)