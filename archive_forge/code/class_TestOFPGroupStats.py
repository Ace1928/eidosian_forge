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
class TestOFPGroupStats(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPGroupStats
    """
    length = ofproto.OFP_GROUP_STATS_SIZE + ofproto.OFP_BUCKET_COUNTER_SIZE
    group_id = 6606
    ref_count = 2102
    packet_count = 6489108735192644493
    byte_count = 7334344481123449724
    buck_packet_count = 3519264449364891087
    buck_byte_count = 3123449724733434448
    bucket_counters = [OFPBucketCounter(buck_packet_count, buck_byte_count)]
    buf_bucket_counters = pack(ofproto.OFP_BUCKET_COUNTER_PACK_STR, buck_packet_count, buck_byte_count)
    fmt = ofproto.OFP_GROUP_STATS_PACK_STR
    buf = pack(fmt, length, group_id, ref_count, packet_count, byte_count) + buf_bucket_counters

    def test_init(self):
        c = OFPGroupStats(self.group_id, self.ref_count, self.packet_count, self.byte_count, self.bucket_counters)
        self.assertEqual(self.group_id, c.group_id)
        self.assertEqual(self.ref_count, c.ref_count)
        self.assertEqual(self.packet_count, c.packet_count)
        self.assertEqual(self.byte_count, c.byte_count)
        self.assertEqual(self.bucket_counters, c.bucket_counters)

    def _test_parser(self, group_id, ref_count, packet_count, byte_count, bucket_counter_cnt):
        length = ofproto.OFP_GROUP_STATS_SIZE + ofproto.OFP_BUCKET_COUNTER_SIZE * bucket_counter_cnt
        fmt = ofproto.OFP_GROUP_STATS_PACK_STR
        buf = pack(fmt, length, group_id, ref_count, packet_count, byte_count)
        bucket_counters = []
        for b in range(bucket_counter_cnt):
            buck_packet_count = b
            buck_byte_count = b
            bucket_counter = OFPBucketCounter(buck_packet_count, buck_byte_count)
            bucket_counters.append(bucket_counter)
            buf_bucket_counters = pack(ofproto.OFP_BUCKET_COUNTER_PACK_STR, buck_packet_count, buck_byte_count)
            buf += buf_bucket_counters
        res = OFPGroupStats.parser(buf, 0)
        self.assertEqual(length, res.length)
        self.assertEqual(group_id, res.group_id)
        self.assertEqual(ref_count, res.ref_count)
        self.assertEqual(packet_count, res.packet_count)
        self.assertEqual(byte_count, res.byte_count)
        for b in range(bucket_counter_cnt):
            self.assertEqual(bucket_counters[b].packet_count, res.bucket_counters[b].packet_count)
            self.assertEqual(bucket_counters[b].byte_count, res.bucket_counters[b].byte_count)

    def test_parser_mid(self):
        bucket_counter_cnt = 2046
        self._test_parser(self.group_id, self.ref_count, self.packet_count, self.byte_count, bucket_counter_cnt)

    def test_parser_max(self):
        group_id = 4294967295
        ref_count = 4294967295
        packet_count = 18446744073709551615
        byte_count = 18446744073709551615
        bucket_counter_cnt = 4093
        self._test_parser(group_id, ref_count, packet_count, byte_count, bucket_counter_cnt)

    def test_parser_min(self):
        group_id = 0
        ref_count = 0
        packet_count = 0
        byte_count = 0
        bucket_counter_cnt = 0
        self._test_parser(group_id, ref_count, packet_count, byte_count, bucket_counter_cnt)