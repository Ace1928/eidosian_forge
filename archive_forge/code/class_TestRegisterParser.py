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
class TestRegisterParser(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser._register_parser
    """

    class _OFPDummy(MsgBase):

        def __init__(self, datapath):
            self.dummy = 'dummy'

        def parser(self):
            return self.dummy

    def test_cls_msg_type(self):
        msg_type = 255
        cls = self._OFPDummy(_Datapath)
        cls.cls_msg_type = msg_type
        res = ofproto_v1_2_parser._register_parser(cls)
        res_parser = ofproto_v1_2_parser._MSG_PARSERS[msg_type]
        del ofproto_v1_2_parser._MSG_PARSERS[msg_type]
        self.assertEqual(res.cls_msg_type, msg_type)
        self.assertTrue(res.dummy)
        self.assertEqual(res_parser(), 'dummy')

    def test_cls_msg_type_none(self):
        cls = OFPHello(_Datapath)
        cls.cls_msg_type = None
        self.assertRaises(AssertionError, ofproto_v1_2_parser._register_parser, cls)

    def test_cls_msg_type_already_registed(self):
        cls = OFPHello(_Datapath)
        self.assertRaises(AssertionError, ofproto_v1_2_parser._register_parser, cls)