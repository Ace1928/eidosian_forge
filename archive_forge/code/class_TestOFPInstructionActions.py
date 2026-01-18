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
class TestOFPInstructionActions(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPInstructionActions
    """
    type_ = ofproto.OFPIT_WRITE_ACTIONS
    len_ = ofproto.OFP_INSTRUCTION_ACTIONS_SIZE + ofproto.OFP_ACTION_OUTPUT_SIZE
    fmt = ofproto.OFP_INSTRUCTION_ACTIONS_PACK_STR
    buf = pack(fmt, type_, len_)
    port = 10976
    max_len = ofproto.OFP_ACTION_OUTPUT_SIZE
    actions = [OFPActionOutput(port, max_len)]
    buf_actions = bytearray()
    actions[0].serialize(buf_actions, 0)
    buf += bytes(buf_actions)

    def test_init(self):
        c = OFPInstructionActions(self.type_, self.actions)
        self.assertEqual(self.type_, c.type)
        self.assertEqual(self.actions, c.actions)

    def _test_parser(self, action_cnt):
        len_ = ofproto.OFP_INSTRUCTION_ACTIONS_SIZE + ofproto.OFP_ACTION_OUTPUT_SIZE * action_cnt
        fmt = ofproto.OFP_INSTRUCTION_ACTIONS_PACK_STR
        buf = pack(fmt, self.type_, len_)
        actions = []
        for a in range(action_cnt):
            port = a
            action = OFPActionOutput(port, self.max_len)
            actions.append(action)
            buf_actions = bytearray()
            actions[a].serialize(buf_actions, 0)
            buf += bytes(buf_actions)
        res = OFPInstructionActions.parser(buf, 0)
        self.assertEqual(res.len, len_)
        self.assertEqual(res.type, self.type_)
        for a in range(action_cnt):
            self.assertEqual(res.actions[a].type, actions[a].type)
            self.assertEqual(res.actions[a].len, actions[a].len)
            self.assertEqual(res.actions[a].port, actions[a].port)
            self.assertEqual(res.actions[a].max_len, actions[a].max_len)

    def test_parser_mid(self):
        self._test_parser(2047)

    def test_parser_max(self):
        self._test_parser(4095)

    def test_parser_min(self):
        self._test_parser(0)

    def _test_serialize(self, action_cnt):
        len_ = ofproto.OFP_INSTRUCTION_ACTIONS_SIZE + ofproto.OFP_ACTION_OUTPUT_SIZE * action_cnt
        actions = []
        for a in range(action_cnt):
            port = a
            action = OFPActionOutput(port, self.max_len)
            actions.append(action)
        c = OFPInstructionActions(self.type_, actions)
        buf = bytearray()
        c.serialize(buf, 0)
        fmt = '!' + ofproto.OFP_INSTRUCTION_ACTIONS_PACK_STR.replace('!', '')
        for a in range(action_cnt):
            fmt += ofproto.OFP_ACTION_OUTPUT_PACK_STR.replace('!', '')
        res = struct.unpack(fmt, bytes(buf))
        self.assertEqual(res[0], self.type_)
        self.assertEqual(res[1], len_)
        for a in range(action_cnt):
            d = 2 + a * 4
            self.assertEqual(res[d], actions[a].type)
            self.assertEqual(res[d + 1], actions[a].len)
            self.assertEqual(res[d + 2], actions[a].port)
            self.assertEqual(res[d + 3], actions[a].max_len)

    def test_serialize_mid(self):
        self._test_serialize(2047)

    def test_serialize_max(self):
        self._test_serialize(4095)

    def test_serialize_min(self):
        self._test_serialize(0)