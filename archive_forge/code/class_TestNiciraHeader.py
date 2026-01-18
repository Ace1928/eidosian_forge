import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
class TestNiciraHeader(unittest.TestCase):
    """ Test case for ofproto_v1_0_parser.NiciraHeader
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        subtype = ofproto.NXT_FLOW_MOD_TABLE_ID
        c = NiciraHeader(object, subtype)
        self.assertEqual(subtype, c.subtype)

    def test_parser(self):
        pass

    def test_serialize(self):

        class Datapath(object):
            ofproto = ofproto
            ofproto_parser = ofproto_v1_0_parser
        data = b'Reply Message.'
        subtype = ofproto.NXT_FLOW_MOD_TABLE_ID
        c = NiciraHeader(Datapath, subtype)
        c.data = data
        c.serialize()
        self.assertEqual(ofproto.OFP_VERSION, c.version)
        self.assertEqual(ofproto.OFPT_VENDOR, c.msg_type)
        self.assertEqual(0, c.xid)
        self.assertEqual(ofproto_common.NX_EXPERIMENTER_ID, c.vendor)
        fmt = '!' + ofproto.OFP_HEADER_PACK_STR.replace('!', '') + ofproto.NICIRA_HEADER_PACK_STR.replace('!', '') + str(len(data)) + 's'
        res = struct.unpack(fmt, bytes(c.buf))
        self.assertEqual(ofproto.OFP_VERSION, res[0])
        self.assertEqual(ofproto.OFPT_VENDOR, res[1])
        self.assertEqual(len(c.buf), res[2])
        self.assertEqual(0, res[3])
        self.assertEqual(ofproto_common.NX_EXPERIMENTER_ID, res[4])
        self.assertEqual(subtype, res[5])
        self.assertEqual(data, res[6])