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
class TestOFPPacketIn(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPPacketIn
    """

    def _test_parser(self, xid, buffer_id, total_len=0, reason=0, table_id=0, data=None):
        if data is None:
            data = b''
        version = ofproto.OFP_VERSION
        msg_type = ofproto.OFPT_PACKET_IN
        msg_len = ofproto.OFP_PACKET_IN_SIZE + len(data)
        fmt = ofproto.OFP_HEADER_PACK_STR
        buf = pack(fmt, version, msg_type, msg_len, xid)
        fmt = ofproto.OFP_PACKET_IN_PACK_STR
        buf += pack(fmt, buffer_id, total_len, reason, table_id)
        buf_match = bytearray()
        match = OFPMatch()
        match.serialize(buf_match, 0)
        buf += bytes(buf_match)
        buf += b'\x00' * 2
        buf += data
        res = OFPPacketIn.parser(object, version, msg_type, msg_len, xid, buf)
        self.assertEqual(version, res.version)
        self.assertEqual(msg_type, res.msg_type)
        self.assertEqual(msg_len, res.msg_len)
        self.assertEqual(xid, res.xid)
        self.assertEqual(buffer_id, res.buffer_id)
        self.assertEqual(total_len, res.total_len)
        self.assertEqual(reason, res.reason)
        self.assertEqual(table_id, res.table_id)
        self.assertTrue(hasattr(res, 'match'))
        self.assertEqual(ofproto.OFPMT_OXM, res.match.type)
        if data:
            self.assertEqual(data[:total_len], res.data)

    def test_data_is_total_len(self):
        xid = 3423224276
        buffer_id = 2926809324
        reason = 128
        table_id = 3
        data = b'PacketIn'
        total_len = len(data)
        self._test_parser(xid, buffer_id, total_len, reason, table_id, data)

    def test_data_is_not_total_len(self):
        xid = 3423224276
        buffer_id = 2926809324
        reason = 128
        table_id = 3
        data = b'PacketIn'
        total_len = len(data) - 1
        self._test_parser(xid, buffer_id, total_len, reason, table_id, data)

    def test_parser_max(self):
        xid = 4294967295
        buffer_id = 4294967295
        reason = 255
        table_id = 255
        data = b'data'.ljust(65511)
        total_len = len(data)
        self._test_parser(xid, buffer_id, total_len, reason, table_id, data)

    def test_parser_min(self):
        xid = 0
        buffer_id = 0
        reason = ofproto.OFPR_NO_MATCH
        table_id = 0
        total_len = 0
        self._test_parser(xid, buffer_id, total_len, reason, table_id)

    def test_parser_p1(self):
        data = b'data'.ljust(8)
        xid = 3423224276
        buffer_id = 2926809324
        total_len = len(data)
        reason = ofproto.OFPR_ACTION
        table_id = 3
        self._test_parser(xid, buffer_id, total_len, reason, table_id, data)

    def test_parser_p2(self):
        data = b'data'.ljust(8)
        xid = 3423224276
        buffer_id = 2926809324
        total_len = len(data)
        reason = ofproto.OFPR_INVALID_TTL
        table_id = 3
        self._test_parser(xid, buffer_id, total_len, reason, table_id, data)