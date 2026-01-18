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
class TestOFPSetConfig(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPSetConfig
    """

    def test_init(self):
        flags = 41186
        miss_send_len = 13838
        c = OFPSetConfig(_Datapath, flags, miss_send_len)
        self.assertEqual(flags, c.flags)
        self.assertEqual(miss_send_len, c.miss_send_len)

    def _test_serialize(self, flags, miss_send_len):
        c = OFPSetConfig(_Datapath, flags, miss_send_len)
        c.serialize()
        self.assertEqual(ofproto.OFP_VERSION, c.version)
        self.assertEqual(ofproto.OFPT_SET_CONFIG, c.msg_type)
        self.assertEqual(0, c.xid)
        fmt = '!' + ofproto.OFP_HEADER_PACK_STR.replace('!', '') + ofproto.OFP_SWITCH_CONFIG_PACK_STR.replace('!', '')
        res = struct.unpack(fmt, bytes(c.buf))
        self.assertEqual(res[0], ofproto.OFP_VERSION)
        self.assertEqual(res[1], ofproto.OFPT_SET_CONFIG)
        self.assertEqual(res[2], len(c.buf))
        self.assertEqual(res[3], 0)
        self.assertEqual(res[4], flags)
        self.assertEqual(res[5], miss_send_len)

    def test_serialize_mid(self):
        flags = 41186
        miss_send_len = 13838
        self._test_serialize(flags, miss_send_len)

    def test_serialize_max(self):
        flags = 65535
        miss_send_len = 65535
        self._test_serialize(flags, miss_send_len)

    def test_serialize_min(self):
        flags = ofproto.OFPC_FRAG_NORMAL
        miss_send_len = 0
        self._test_serialize(flags, miss_send_len)

    def test_serialize_p1(self):
        flags = ofproto.OFPC_FRAG_DROP
        miss_send_len = 13838
        self._test_serialize(flags, miss_send_len)

    def test_serialize_p2(self):
        flags = ofproto.OFPC_FRAG_REASM
        miss_send_len = 13838
        self._test_serialize(flags, miss_send_len)

    def test_serialize_p3(self):
        flags = ofproto.OFPC_FRAG_MASK
        miss_send_len = 13838
        self._test_serialize(flags, miss_send_len)

    def test_serialize_p4(self):
        flags = ofproto.OFPC_INVALID_TTL_TO_CONTROLLER
        miss_send_len = 13838
        self._test_serialize(flags, miss_send_len)

    def test_serialize_check_flags(self):
        flags = None
        miss_send_len = 13838
        c = OFPSetConfig(_Datapath, flags, miss_send_len)
        self.assertRaises(AssertionError, c.serialize)

    def test_serialize_check_miss_send_len(self):
        flags = 41186
        miss_send_len = None
        c = OFPSetConfig(_Datapath, flags, miss_send_len)
        self.assertRaises(AssertionError, c.serialize)