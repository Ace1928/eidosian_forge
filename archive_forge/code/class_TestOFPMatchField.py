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
class TestOFPMatchField(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPMatchField
    """

    def test_init_hasmask_true(self):
        header = 256
        res = OFPMatchField(header)
        self.assertEqual(res.header, header)
        self.assertEqual(res.n_bytes, (header & 255) // 2)
        self.assertEqual(res.length, 0)

    def test_init_hasmask_false(self):
        header = 0
        res = OFPMatchField(header)
        self.assertEqual(res.header, header)
        self.assertEqual(res.n_bytes, header & 255)
        self.assertEqual(res.length, 0)