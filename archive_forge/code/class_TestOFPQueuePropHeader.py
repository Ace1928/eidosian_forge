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
class TestOFPQueuePropHeader(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPQueuePropHeader
    """
    property_ = 1
    len_ = 10

    def test_init(self):
        c = OFPQueuePropHeader(self.property_, self.len_)
        self.assertEqual(self.property_, c.property)
        self.assertEqual(self.len_, c.len)

    def _test_serialize(self, property_, len_):
        c = OFPQueuePropHeader(property_, len_)
        buf = bytearray()
        c.serialize(buf, 0)
        fmt = ofproto.OFP_QUEUE_PROP_HEADER_PACK_STR
        res = struct.unpack(fmt, bytes(buf))
        self.assertEqual(res[0], property_)
        self.assertEqual(res[1], len_)

    def test_serialize_mid(self):
        self._test_serialize(self.property_, self.len_)

    def test_serialize_max(self):
        self._test_serialize(65535, 65535)

    def test_serialize_min(self):
        self._test_serialize(0, 0)