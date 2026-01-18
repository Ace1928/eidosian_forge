import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
class TestOFPVendorStatsRequest(unittest.TestCase):
    """ Test case for ofproto_v1_0_parser.OFPVendorStatsRequest
    """

    class Datapath(object):
        ofproto = ofproto
        ofproto_parser = ofproto_v1_0_parser
    flags = {'buf': b'\x00\x00', 'val': 0}
    vendor = {'buf': b'\xff\xff\xff\xff', 'val': ofproto.OFPAT_VENDOR}
    specific_data = b'specific_data'
    c = OFPVendorStatsRequest(Datapath, flags['val'], vendor['val'], specific_data)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(ofproto.OFPST_VENDOR, self.c.type)
        self.assertEqual(self.flags['val'], self.c.flags)
        self.assertEqual(self.vendor['val'], self.c.vendor)
        self.assertEqual(self.specific_data, self.c.specific_data)

    def test_parser(self):
        pass

    def test_serialize(self):
        self.c.serialize()
        self.assertEqual(ofproto.OFP_VERSION, self.c.version)
        self.assertEqual(ofproto.OFPT_STATS_REQUEST, self.c.msg_type)
        self.assertEqual(0, self.c.xid)
        fmt = '!' + ofproto.OFP_HEADER_PACK_STR.replace('!', '') + ofproto.OFP_STATS_MSG_PACK_STR.replace('!', '') + ofproto.OFP_VENDOR_STATS_MSG_PACK_STR.replace('!', '') + str(len(self.specific_data)) + 's'
        res = struct.unpack(fmt, bytes(self.c.buf))
        self.assertEqual(ofproto.OFP_VERSION, res[0])
        self.assertEqual(ofproto.OFPT_STATS_REQUEST, res[1])
        self.assertEqual(len(self.c.buf), res[2])
        self.assertEqual(0, res[3])
        self.assertEqual(ofproto.OFPST_VENDOR, res[4])
        self.assertEqual(self.flags['val'], res[5])
        self.assertEqual(self.vendor['val'], res[6])
        self.assertEqual(self.specific_data, res[7])