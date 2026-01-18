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
class TestOFPQueueGetConfigRequest(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPQueueGetConfigRequest
    """
    port = 41186

    def test_init(self):
        c = OFPQueueGetConfigRequest(_Datapath, self.port)
        self.assertEqual(self.port, c.port)

    def _test_serialize(self, port):
        c = OFPQueueGetConfigRequest(_Datapath, port)
        c.serialize()
        self.assertEqual(ofproto.OFP_VERSION, c.version)
        self.assertEqual(ofproto.OFPT_QUEUE_GET_CONFIG_REQUEST, c.msg_type)
        self.assertEqual(0, c.xid)
        fmt = ofproto.OFP_HEADER_PACK_STR + ofproto.OFP_QUEUE_GET_CONFIG_REQUEST_PACK_STR[1:]
        res = struct.unpack(fmt, bytes(c.buf))
        self.assertEqual(res[0], ofproto.OFP_VERSION)
        self.assertEqual(res[1], ofproto.OFPT_QUEUE_GET_CONFIG_REQUEST)
        self.assertEqual(res[2], len(c.buf))
        self.assertEqual(res[3], 0)
        self.assertEqual(res[4], port)

    def test_serialize_mid(self):
        self._test_serialize(self.port)

    def test_serialize_max(self):
        self._test_serialize(4294967295)

    def test_serialize_min(self):
        self._test_serialize(0)

    def test_serialize_p1(self):
        self._test_serialize(ofproto.OFPP_MAX)