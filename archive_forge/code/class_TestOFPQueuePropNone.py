import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
class TestOFPQueuePropNone(unittest.TestCase):
    """ Test case for ofproto_v1_0_parser.OFPQueuePropNone
    """
    property = {'buf': b'\x00\x00', 'val': ofproto.OFPQT_NONE}
    len = {'buf': b'\x00\x08', 'val': ofproto.OFP_QUEUE_PROP_HEADER_SIZE}
    zfill = b'\x00' * 4
    c = OFPQueuePropNone()

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        cls = OFPQueuePropHeader._QUEUE_PROPERTIES[self.c.cls_prop_type]
        self.assertEqual(self.property['val'], self.c.cls_prop_type)
        self.assertEqual(self.property['val'], self.c.property)
        self.assertEqual(self.property['val'], cls.cls_prop_type)
        self.assertEqual(self.len['val'], self.c.cls_prop_len)
        self.assertEqual(self.len['val'], self.c.len)
        self.assertEqual(self.len['val'], cls.cls_prop_len)

    def test_parser(self):
        buf = self.property['buf'] + self.len['buf'] + self.zfill
        self.assertTrue(self.c.parser(buf, 0))