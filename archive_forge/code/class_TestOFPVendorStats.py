import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
class TestOFPVendorStats(unittest.TestCase):
    """ Test case for ofproto_v1_0_parser.OFPVendorStats
    """
    specific_data = 'specific_data'
    specific_data_after = 'data'
    offset = specific_data.find(specific_data_after)
    c = OFPVendorStats(specific_data)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.specific_data, self.c.specific_data)

    def test_parser(self):
        res = self.c.parser(self.specific_data, self.offset)
        self.assertEqual(self.specific_data_after, res.specific_data)