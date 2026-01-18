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
class TestOFPQueuePropMaxRate(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPQueuePropMaxRate
    """
    rate = 100
    buf = pack(ofproto.OFP_QUEUE_PROP_MAX_RATE_PACK_STR, rate)
    c = OFPQueuePropMaxRate(rate)

    def _test_parser(self, rate):
        buf = pack(ofproto.OFP_QUEUE_PROP_MAX_RATE_PACK_STR, rate)
        res = OFPQueuePropMaxRate.parser(buf, 0)
        self.assertEqual(rate, res.rate)

    def test_parser_mid(self):
        self._test_parser(100)

    def test_parser_max(self):
        self._test_parser(65535)

    def test_parser_min(self):
        self._test_parser(0)