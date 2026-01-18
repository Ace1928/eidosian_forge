import os
import socket
import sys
import unittest
from unittest import mock
from os_ken.lib import pcaplib
from os_ken.lib.packet import packet
from os_ken.lib.packet import zebra
from os_ken.utils import binary_str
class TestZebraBfdDestinationDeregisterMultiHopDisabled(unittest.TestCase):
    buf = b'\x00\x00\x00\x01\x00\x02\xc0\xa8\x01\x01\x00\x00\x02\xc0\xa8\x01\x02\x04eth0'
    pid = 1
    dst_family = socket.AF_INET
    dst_prefix = '192.168.1.1'
    multi_hop = 0
    src_family = socket.AF_INET
    src_prefix = '192.168.1.2'
    multi_hop_count = None
    ifname = 'eth0'

    @_patch_frr_v2
    def test_parser(self):
        body = zebra.ZebraBfdDestinationDeregister.parse(self.buf)
        self.assertEqual(self.pid, body.pid)
        self.assertEqual(self.dst_family, body.dst_family)
        self.assertEqual(self.dst_prefix, body.dst_prefix)
        self.assertEqual(self.multi_hop, body.multi_hop)
        self.assertEqual(self.src_family, body.src_family)
        self.assertEqual(self.src_prefix, body.src_prefix)
        self.assertEqual(self.multi_hop_count, body.multi_hop_count)
        self.assertEqual(self.ifname, body.ifname)
        buf = body.serialize()
        self.assertEqual(binary_str(self.buf), binary_str(buf))