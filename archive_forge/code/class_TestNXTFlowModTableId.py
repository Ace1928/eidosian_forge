import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
class TestNXTFlowModTableId(unittest.TestCase):
    """ Test case for ofproto_v1_0_parser.NXTFlowModTableId
    """
    set_ = {'buf': b'q', 'val': 113}
    zfill = b'\x00' * 7

    class Datapath(object):
        ofproto = ofproto
        ofproto_parser = ofproto_v1_0_parser
    c = NXTFlowModTableId(Datapath, set_['val'])

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.set_['val'], self.c.set)

    def test_parser(self):
        pass

    def test_serialize(self):
        self.c.serialize()
        self.assertEqual(ofproto.OFP_VERSION, self.c.version)
        self.assertEqual(ofproto.OFPT_VENDOR, self.c.msg_type)
        self.assertEqual(0, self.c.xid)
        self.assertEqual(ofproto_common.NX_EXPERIMENTER_ID, self.c.vendor)
        fmt = '!' + ofproto.OFP_HEADER_PACK_STR.replace('!', '') + ofproto.NICIRA_HEADER_PACK_STR.replace('!', '') + ofproto.NX_FLOW_MOD_TABLE_ID_PACK_STR.replace('!', '')
        res = struct.unpack(fmt, bytes(self.c.buf))
        self.assertEqual(ofproto.OFP_VERSION, res[0])
        self.assertEqual(ofproto.OFPT_VENDOR, res[1])
        self.assertEqual(len(self.c.buf), res[2])
        self.assertEqual(0, res[3])
        self.assertEqual(ofproto_common.NX_EXPERIMENTER_ID, res[4])
        self.assertEqual(ofproto.NXT_FLOW_MOD_TABLE_ID, res[5])
        self.assertEqual(self.set_['val'], res[6])