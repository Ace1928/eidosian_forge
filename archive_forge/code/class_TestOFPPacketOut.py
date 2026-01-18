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
class TestOFPPacketOut(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPPacketOut
    """

    def _test_init(self, in_port):
        buffer_id = 4294967295
        data = b'Message'
        out_port = 10976
        actions = [OFPActionOutput(out_port, 0)]
        c = OFPPacketOut(_Datapath, buffer_id, in_port, actions, data)
        self.assertEqual(buffer_id, c.buffer_id)
        self.assertEqual(in_port, c.in_port)
        self.assertEqual(0, c.actions_len)
        self.assertEqual(data, c.data)
        self.assertEqual(actions, c.actions)

    def test_init(self):
        in_port = 263253
        self._test_init(in_port)

    def test_init_check_in_port(self):
        in_port = None
        self.assertRaises(AssertionError, self._test_init, in_port)

    def _test_serialize(self, buffer_id, in_port, action_cnt=0, data=None):
        actions = []
        for i in range(action_cnt):
            actions.append(ofproto_v1_2_parser.OFPActionOutput(i, 0))
        c = OFPPacketOut(_Datapath, buffer_id, in_port, actions, data)
        c.serialize()
        self.assertEqual(ofproto.OFP_VERSION, c.version)
        self.assertEqual(ofproto.OFPT_PACKET_OUT, c.msg_type)
        self.assertEqual(0, c.xid)
        fmt = ofproto.OFP_HEADER_PACK_STR + ofproto.OFP_PACKET_OUT_PACK_STR[1:] + ofproto.OFP_ACTION_OUTPUT_PACK_STR[1:] * action_cnt
        if data is not None:
            fmt += str(len(data)) + 's'
        res = struct.unpack(fmt, bytes(c.buf))
        self.assertEqual(res[0], ofproto.OFP_VERSION)
        self.assertEqual(res[1], ofproto.OFPT_PACKET_OUT)
        self.assertEqual(res[2], len(c.buf))
        self.assertEqual(res[3], 0)
        self.assertEqual(res[4], buffer_id)
        self.assertEqual(res[5], in_port)
        self.assertEqual(res[6], ofproto.OFP_ACTION_OUTPUT_SIZE * action_cnt)
        for i in range(action_cnt):
            index = 7 + i * 4
            self.assertEqual(res[index], ofproto.OFPAT_OUTPUT)
            self.assertEqual(res[index + 1], ofproto.OFP_ACTION_OUTPUT_SIZE)
            self.assertEqual(res[index + 2], i)
            self.assertEqual(res[index + 3], 0)
        if data:
            self.assertEqual(res[-1], data)

    def test_serialize_true(self):
        buffer_id = 4294967295
        in_port = 263253
        action_cnt = 2
        data = b'Message'
        self._test_serialize(buffer_id, in_port, action_cnt, data)

    def test_serialize_none(self):
        buffer_id = 4294967295
        in_port = 263253
        self._test_serialize(buffer_id, in_port)

    def test_serialize_max(self):
        buffer_id = 4294967295
        in_port = 4294967295
        action_cnt = 1
        data = b'Message'.ljust(65495)
        self._test_serialize(buffer_id, in_port, action_cnt, data)

    def test_serialize_min(self):
        buffer_id = 0
        in_port = 0
        self._test_serialize(buffer_id, in_port)

    def test_serialize_p1(self):
        buffer_id = 2147483648
        in_port = ofproto.OFPP_CONTROLLER
        self._test_serialize(buffer_id, in_port)

    def test_serialize_check_buffer_id(self):
        buffer_id = 2147483648
        in_port = 1
        action_cnt = 0
        data = b'DATA'
        self.assertRaises(AssertionError, self._test_serialize, buffer_id, in_port, action_cnt, data)