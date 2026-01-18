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
class TestOFPFlowStats(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPFlowStats
    """

    def test_init(self):
        length = ofproto.OFP_FLOW_STATS_SIZE
        table_id = 81
        duration_sec = 2484712402
        duration_nsec = 3999715196
        priority = 57792
        idle_timeout = 36368
        hard_timeout = 54425
        cookie = 793171083674290912
        packet_count = 5142202600015232219
        byte_count = 2659740543924820419
        match = OFPMatch()
        in_port = 2
        match.set_in_port(in_port)
        goto_table = 3
        instructions = [OFPInstructionGotoTable(goto_table)]
        c = OFPFlowStats(table_id, duration_sec, duration_nsec, priority, idle_timeout, hard_timeout, cookie, packet_count, byte_count, match, instructions)
        self.assertEqual(table_id, c.table_id)
        self.assertEqual(duration_sec, c.duration_sec)
        self.assertEqual(duration_nsec, c.duration_nsec)
        self.assertEqual(priority, c.priority)
        self.assertEqual(idle_timeout, c.idle_timeout)
        self.assertEqual(hard_timeout, c.hard_timeout)
        self.assertEqual(cookie, c.cookie)
        self.assertEqual(packet_count, c.packet_count)
        self.assertEqual(byte_count, c.byte_count)
        self.assertEqual(in_port, c.match._flow.in_port)
        self.assertEqual(goto_table, c.instructions[0].table_id)

    def _test_parser(self, table_id, duration_sec, duration_nsec, priority, idle_timeout, hard_timeout, cookie, packet_count, byte_count, inst_cnt=0):
        length = ofproto.OFP_FLOW_STATS_SIZE + calcsize(MTEthType.pack_str[1:] + '6x') + ofproto.OFP_INSTRUCTION_GOTO_TABLE_SIZE * inst_cnt
        buf = pack(ofproto.OFP_FLOW_STATS_PACK_STR, length, table_id, duration_sec, duration_nsec, priority, idle_timeout, hard_timeout, cookie, packet_count, byte_count)
        match = OFPMatch()
        dl_type = 2048
        match.set_dl_type(dl_type)
        match_buf = bytearray()
        match.serialize(match_buf, 0)
        buf += bytes(match_buf)
        for i in range(inst_cnt):
            inst = OFPInstructionGotoTable(1)
            inst_buf = bytearray()
            inst.serialize(inst_buf, 0)
            buf += bytes(inst_buf)
        res = OFPFlowStats.parser(buf, 0)
        self.assertEqual(length, res.length)
        self.assertEqual(table_id, res.table_id)
        self.assertEqual(duration_sec, res.duration_sec)
        self.assertEqual(duration_nsec, res.duration_nsec)
        self.assertEqual(priority, res.priority)
        self.assertEqual(idle_timeout, res.idle_timeout)
        self.assertEqual(hard_timeout, res.hard_timeout)
        self.assertEqual(cookie, res.cookie)
        self.assertEqual(packet_count, res.packet_count)
        self.assertEqual(byte_count, res.byte_count)
        self.assertEqual(dl_type, res.match.fields[0].value)
        for i in range(inst_cnt):
            self.assertEqual(1, res.instructions[i].table_id)

    def test_parser_mid(self):
        table_id = 81
        duration_sec = 2484712402
        duration_nsec = 3999715196
        priority = 57792
        idle_timeout = 36368
        hard_timeout = 54425
        cookie = 793171083674290912
        packet_count = 5142202600015232219
        byte_count = 2659740543924820419
        inst_cnt = 2
        self._test_parser(table_id, duration_sec, duration_nsec, priority, idle_timeout, hard_timeout, cookie, packet_count, byte_count, inst_cnt)

    def test_parser_max(self):
        table_id = 255
        duration_sec = 65535
        duration_nsec = 65535
        priority = 65535
        idle_timeout = 255
        hard_timeout = 255
        cookie = 18446744073709551615
        packet_count = 18446744073709551615
        byte_count = 18446744073709551615
        inst_cnt = 8183
        self._test_parser(table_id, duration_sec, duration_nsec, priority, idle_timeout, hard_timeout, cookie, packet_count, byte_count, inst_cnt)

    def test_parser_min(self):
        self._test_parser(0, 0, 0, 0, 0, 0, 0, 0, 0)