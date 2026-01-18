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
class TestOFPBarrierRequest(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPBarrierRequest
    """

    def test_serialize(self):
        c = OFPBarrierRequest(_Datapath)
        c.serialize()
        self.assertEqual(ofproto.OFP_VERSION, c.version)
        self.assertEqual(ofproto.OFPT_BARRIER_REQUEST, c.msg_type)
        self.assertEqual(ofproto.OFP_HEADER_SIZE, c.msg_len)
        self.assertEqual(0, c.xid)
        fmt = ofproto.OFP_HEADER_PACK_STR
        res = unpack(fmt, bytes(c.buf))
        self.assertEqual(ofproto.OFP_VERSION, res[0])
        self.assertEqual(ofproto.OFPT_BARRIER_REQUEST, res[1])
        self.assertEqual(len(c.buf), res[2])
        self.assertEqual(0, c.xid)