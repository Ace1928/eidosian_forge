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
class TestOFPActionExperimenter(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPActionExperimenter
    """
    type_ = ofproto.OFPAT_EXPERIMENTER
    len_ = ofproto.OFP_ACTION_EXPERIMENTER_HEADER_SIZE
    fmt = ofproto.OFP_ACTION_EXPERIMENTER_HEADER_PACK_STR

    def test_init(self):
        experimenter = 4294967295
        c = OFPActionExperimenter(experimenter)
        self.assertEqual(experimenter, c.experimenter)

    def _test_parser(self, experimenter):
        buf = pack(self.fmt, self.type_, self.len_, experimenter)
        res = OFPActionExperimenter.parser(buf, 0)
        self.assertEqual(res.type, self.type_)
        self.assertEqual(res.len, self.len_)
        self.assertEqual(res.experimenter, experimenter)

    def test_parser_mid(self):
        experimenter = 2147483648
        self._test_parser(experimenter)

    def test_parser_max(self):
        experimenter = 4294967295
        self._test_parser(experimenter)

    def test_parser_min(self):
        experimenter = 0
        self._test_parser(experimenter)

    def _test_serialize(self, experimenter):
        c = OFPActionExperimenter(experimenter)
        buf = bytearray()
        c.serialize(buf, 0)
        res = struct.unpack(self.fmt, bytes(buf))
        self.assertEqual(res[0], self.type_)
        self.assertEqual(res[1], self.len_)
        self.assertEqual(res[2], experimenter)

    def test_serialize_mid(self):
        experimenter = 2147483648
        self._test_serialize(experimenter)

    def test_serialize_max(self):
        experimenter = 4294967295
        self._test_serialize(experimenter)

    def test_serialize_min(self):
        experimenter = 0
        self._test_serialize(experimenter)