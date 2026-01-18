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
def _test_msg_parser(self, xid, msg_len):
    version = ofproto.OFP_VERSION
    msg_type = ofproto.OFPT_HELLO
    fmt = ofproto.OFP_HEADER_PACK_STR
    buf = pack(fmt, version, msg_type, msg_len, xid)
    c = msg_parser(_Datapath, version, msg_type, msg_len, xid, buf)
    self.assertEqual(version, c.version)
    self.assertEqual(msg_type, c.msg_type)
    self.assertEqual(msg_len, c.msg_len)
    self.assertEqual(xid, c.xid)
    fmt = ofproto.OFP_HEADER_PACK_STR
    res = struct.unpack(fmt, c.buf)
    self.assertEqual(version, res[0])
    self.assertEqual(msg_type, res[1])
    self.assertEqual(msg_len, res[2])
    self.assertEqual(xid, res[3])