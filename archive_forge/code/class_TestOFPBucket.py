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
class TestOFPBucket(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPBucket
    """

    def test_init(self):
        weight = 4386
        watch_port = 6606
        watch_group = 3
        port = 3
        max_len = 1500
        actions = [OFPActionOutput(port, max_len)]
        c = OFPBucket(weight, watch_port, watch_group, actions)
        self.assertEqual(weight, c.weight)
        self.assertEqual(watch_port, c.watch_port)
        self.assertEqual(watch_group, c.watch_group)
        self.assertEqual(1, len(c.actions))
        self.assertEqual(port, c.actions[0].port)
        self.assertEqual(max_len, c.actions[0].max_len)

    def _test_parser(self, weight, watch_port, watch_group, action_cnt):
        len_ = ofproto.OFP_BUCKET_SIZE + ofproto.OFP_ACTION_OUTPUT_SIZE * action_cnt
        fmt = ofproto.OFP_BUCKET_PACK_STR
        buf = pack(fmt, len_, weight, watch_port, watch_group)
        actions = []
        for a in range(action_cnt):
            port = a
            max_len = ofproto.OFP_ACTION_OUTPUT_SIZE
            action = OFPActionOutput(port, max_len)
            actions.append(action)
            buf_actions = bytearray()
            actions[a].serialize(buf_actions, 0)
            buf += bytes(buf_actions)
        res = OFPBucket.parser(buf, 0)
        self.assertEqual(weight, res.weight)
        self.assertEqual(watch_port, res.watch_port)
        self.assertEqual(watch_group, res.watch_group)
        for a in range(action_cnt):
            self.assertEqual(actions[a].type, res.actions[a].type)
            self.assertEqual(actions[a].len, res.actions[a].len)
            self.assertEqual(actions[a].port, res.actions[a].port)
            self.assertEqual(actions[a].max_len, res.actions[a].max_len)

    def test_parser_mid(self):
        weight = 4386
        watch_port = 6606
        watch_group = 3
        action_cnt = 2047
        self._test_parser(weight, watch_port, watch_group, action_cnt)

    def test_parser_max(self):
        weight = 65535
        watch_port = 4294967295
        watch_group = 4294967295
        action_cnt = 4094
        self._test_parser(weight, watch_port, watch_group, action_cnt)

    def test_parser_min(self):
        weight = 0
        watch_port = 0
        watch_group = 0
        action_cnt = 0
        self._test_parser(weight, watch_port, watch_group, action_cnt)

    def _test_serialize(self, weight, watch_port, watch_group, action_cnt):
        len_ = ofproto.OFP_BUCKET_SIZE + ofproto.OFP_ACTION_OUTPUT_SIZE * action_cnt
        actions = []
        for a in range(action_cnt):
            port = a
            max_len = ofproto.OFP_ACTION_OUTPUT_SIZE
            action = OFPActionOutput(port, max_len)
            actions.append(action)
        c = OFPBucket(weight, watch_port, watch_group, actions)
        buf = bytearray()
        c.serialize(buf, 0)
        fmt = ofproto.OFP_BUCKET_PACK_STR
        for a in range(action_cnt):
            fmt += ofproto.OFP_ACTION_OUTPUT_PACK_STR[1:]
        res = struct.unpack(fmt, bytes(buf))
        self.assertEqual(res[0], len_)
        self.assertEqual(res[1], weight)
        self.assertEqual(res[2], watch_port)
        self.assertEqual(res[3], watch_group)
        for a in range(action_cnt):
            d = 4 + a * 4
            self.assertEqual(res[d], actions[a].type)
            self.assertEqual(res[d + 1], actions[a].len)
            self.assertEqual(res[d + 2], actions[a].port)
            self.assertEqual(res[d + 3], actions[a].max_len)

    def test_serialize_mid(self):
        weight = 4386
        watch_port = 6606
        watch_group = 3
        action_cnt = 2047
        self._test_serialize(weight, watch_port, watch_group, action_cnt)

    def test_serialize_max(self):
        weight = 65535
        watch_port = 4294967295
        watch_group = 4294967295
        action_cnt = 4094
        self._test_serialize(weight, watch_port, watch_group, action_cnt)

    def test_serialize_min(self):
        weight = 0
        watch_port = 0
        watch_group = 0
        action_cnt = 0
        self._test_serialize(weight, watch_port, watch_group, action_cnt)