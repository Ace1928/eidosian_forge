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
class TestOFPGroupFeaturesStatsRequest(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPGroupFeaturesStatsRequest
    """

    def test_serialize(self):
        c = OFPGroupFeaturesStatsRequest(_Datapath)
        c.serialize()
        fmt = '!' + ofproto.OFP_HEADER_PACK_STR.replace('!', '') + ofproto.OFP_STATS_REQUEST_PACK_STR.replace('!', '')
        res = struct.unpack(fmt, bytes(c.buf))
        self.assertEqual(res[0], ofproto.OFP_VERSION)
        self.assertEqual(res[1], ofproto.OFPT_STATS_REQUEST)
        self.assertEqual(res[2], len(c.buf))
        self.assertEqual(res[3], 0)
        self.assertEqual(res[4], ofproto.OFPST_GROUP_FEATURES)
        self.assertEqual(res[5], 0)