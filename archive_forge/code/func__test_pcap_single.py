import os
import socket
import sys
import unittest
from unittest import mock
from os_ken.lib import pcaplib
from os_ken.lib.packet import packet
from os_ken.lib.packet import zebra
from os_ken.utils import binary_str
def _test_pcap_single(self, f):
    zebra_pcap_file = os.path.join(PCAP_DATA_DIR, f + '.pcap')
    for _, buf in pcaplib.Reader(open(zebra_pcap_file, 'rb')):
        pkt = packet.Packet(buf)
        zebra_pkts = pkt.get_protocols(zebra.ZebraMessage)
        for zebra_pkt in zebra_pkts:
            self.assertTrue(isinstance(zebra_pkt, zebra.ZebraMessage), 'Failed to parse Zebra message: %s' % pkt)
        self.assertTrue(not isinstance(pkt.protocols[-1], (bytes, bytearray)), 'Some messages could not be parsed in %s: %s' % (f, pkt))
        pkt.serialize()
        self.assertEqual(binary_str(buf), binary_str(pkt.data))