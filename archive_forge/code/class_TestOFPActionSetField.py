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
class TestOFPActionSetField(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPActionSetField
    """
    type_ = ofproto.OFPAT_SET_FIELD
    header = ofproto.OXM_OF_IN_PORT
    in_port = 6606
    field = MTInPort(header, in_port)
    length = ofproto.OFP_ACTION_SET_FIELD_SIZE + field.oxm_len()
    len_ = utils.round_up(length, 8)
    fmt = '!HHII4x'
    buf = pack(fmt, type_, len_, header, in_port)
    c = OFPActionSetField(field)

    def test_init(self):
        self.assertEqual(self.field, self.c.field)

    def test_parser(self):
        res = self.c.parser(self.buf, 0)
        self.assertEqual(res.type, self.type_)
        self.assertEqual(res.len, self.len_)
        self.assertEqual(res.field.header, self.header)
        self.assertEqual(res.field.value, self.in_port)

    def test_serialize(self):
        buf = bytearray()
        self.c.serialize(buf, 0)
        res = struct.unpack(self.fmt, bytes(buf))
        self.assertEqual(res[0], self.type_)
        self.assertEqual(res[1], self.len_)
        self.assertEqual(res[2], self.header)
        self.assertEqual(res[3], self.in_port)