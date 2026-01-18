import os
import socket
import sys
import unittest
from unittest import mock
from os_ken.lib import pcaplib
from os_ken.lib.packet import packet
from os_ken.lib.packet import zebra
from os_ken.utils import binary_str
class TestZebraMplsLabelsAddIPv6(unittest.TestCase):
    buf = b'\t\x00\x00\x00\n \x01\r\xb8\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@ \x01\r\xb8\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x10\x00\x00\x00d\x00\x00\x00\x03'
    route_type = zebra.ZEBRA_ROUTE_BGP
    family = socket.AF_INET6
    prefix = '2001:db8::/64'
    gate_addr = '2001:db8::1'
    distance = 16
    in_label = 100
    out_label = zebra.MPLS_IMP_NULL_LABEL

    @_patch_frr_v2
    def test_parser(self):
        body = zebra.ZebraMplsLabelsAdd.parse(self.buf)
        self.assertEqual(self.route_type, body.route_type)
        self.assertEqual(self.family, body.family)
        self.assertEqual(self.prefix, body.prefix)
        self.assertEqual(self.gate_addr, body.gate_addr)
        self.assertEqual(self.distance, body.distance)
        self.assertEqual(self.in_label, body.in_label)
        self.assertEqual(self.out_label, body.out_label)
        buf = body.serialize()
        self.assertEqual(binary_str(self.buf), binary_str(buf))