import logging
import os
import sys
import unittest
from os_ken.lib import pcaplib
from os_ken.lib.packet import geneve
from os_ken.lib.packet import packet
from os_ken.utils import binary_str
class Test_geneve(unittest.TestCase):
    """
    Test case for os_ken.lib.packet.geneve.
    """

    def test_parser(self):
        files = ['geneve_unknown']
        for f in files:
            for _, buf in pcaplib.Reader(open(GENEVE_DATA_DIR + f + '.pcap', 'rb')):
                pkt = packet.Packet(buf)
                geneve_pkt = pkt.get_protocol(geneve.geneve)
                self.assertTrue(isinstance(geneve_pkt, geneve.geneve), 'Failed to parse Geneve message: %s' % pkt)
                pkt.serialize()
                self.assertEqual(buf, pkt.data, "b'%s' != b'%s'" % (binary_str(buf), binary_str(pkt.data)))