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
class TestOFPQueueStatsRequest(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPQueueStatsRequest
    """
    port_no = 41186
    queue_id = 6606

    def test_init(self):
        c = OFPQueueStatsRequest(_Datapath, self.port_no, self.queue_id)
        self.assertEqual(self.port_no, c.port_no)
        self.assertEqual(self.queue_id, c.queue_id)

    def _test_serialize(self, port_no, queue_id):
        c = OFPQueueStatsRequest(_Datapath, port_no, queue_id)
        c.serialize()
        self.assertEqual(ofproto.OFP_VERSION, c.version)
        self.assertEqual(ofproto.OFPT_STATS_REQUEST, c.msg_type)
        self.assertEqual(0, c.xid)
        fmt = '!' + ofproto.OFP_HEADER_PACK_STR.replace('!', '') + ofproto.OFP_STATS_REQUEST_PACK_STR.replace('!', '') + ofproto.OFP_QUEUE_STATS_REQUEST_PACK_STR.replace('!', '')
        res = struct.unpack(fmt, bytes(c.buf))
        self.assertEqual(res[0], ofproto.OFP_VERSION)
        self.assertEqual(res[1], ofproto.OFPT_STATS_REQUEST)
        self.assertEqual(res[2], len(c.buf))
        self.assertEqual(res[3], 0)
        self.assertEqual(res[4], ofproto.OFPST_QUEUE)
        self.assertEqual(res[5], 0)
        self.assertEqual(res[6], port_no)
        self.assertEqual(res[7], queue_id)

    def test_serialize_mid(self):
        self._test_serialize(self.port_no, self.queue_id)

    def test_serialize_max(self):
        self._test_serialize(4294967295, 4294967295)

    def test_serialize_min(self):
        self._test_serialize(0, 0)

    def test_serialize_p1(self):
        self._test_serialize(ofproto.OFPP_MAX, self.queue_id)

    def test_serialize_p2(self):
        self._test_serialize(ofproto.OFPP_IN_PORT, self.queue_id)

    def test_serialize_p3(self):
        self._test_serialize(ofproto.OFPP_NORMAL, self.queue_id)

    def test_serialize_p4(self):
        self._test_serialize(ofproto.OFPP_TABLE, self.queue_id)

    def test_serialize_p5(self):
        self._test_serialize(ofproto.OFPP_FLOOD, self.queue_id)

    def test_serialize_p6(self):
        self._test_serialize(ofproto.OFPP_ALL, self.queue_id)

    def test_serialize_p7(self):
        self._test_serialize(ofproto.OFPP_CONTROLLER, self.queue_id)

    def test_serialize_p8(self):
        self._test_serialize(ofproto.OFPP_LOCAL, self.queue_id)