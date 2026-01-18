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
class TestOFPRoleReply(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPRoleReply
    """
    role = 2147483648
    generation_id = 1270985291017894273

    def _test_parser(self, role, generation_id):
        version = ofproto.OFP_VERSION
        msg_type = ofproto.OFPT_ROLE_REPLY
        msg_len = ofproto.OFP_ROLE_REQUEST_SIZE
        xid = 2495926989
        fmt = ofproto.OFP_HEADER_PACK_STR
        buf = pack(fmt, version, msg_type, msg_len, xid)
        fmt = ofproto.OFP_ROLE_REQUEST_PACK_STR
        buf += pack(fmt, role, generation_id)
        res = OFPRoleReply.parser(object, version, msg_type, msg_len, xid, buf)
        self.assertEqual(version, res.version)
        self.assertEqual(msg_type, res.msg_type)
        self.assertEqual(msg_len, res.msg_len)
        self.assertEqual(xid, res.xid)
        self.assertEqual(role, res.role)
        self.assertEqual(generation_id, res.generation_id)

    def test_parser_mid(self):
        self._test_parser(self.role, self.generation_id)

    def test_parser_max(self):
        role = 4294967295
        generation_id = 18446744073709551615
        self._test_parser(role, generation_id)

    def test_parser_min(self):
        role = ofproto.OFPCR_ROLE_NOCHANGE
        generation_id = 0
        self._test_parser(role, generation_id)

    def test_parser_p1(self):
        role = ofproto.OFPCR_ROLE_EQUAL
        self._test_parser(role, self.generation_id)

    def test_parser_p2(self):
        role = ofproto.OFPCR_ROLE_MASTER
        self._test_parser(role, self.generation_id)

    def test_parser_p3(self):
        role = ofproto.OFPCR_ROLE_SLAVE
        self._test_parser(role, self.generation_id)