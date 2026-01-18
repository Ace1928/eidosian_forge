import os
import socket
import sys
import unittest
from unittest import mock
from os_ken.lib import pcaplib
from os_ken.lib.packet import packet
from os_ken.lib.packet import zebra
from os_ken.utils import binary_str
class TestZebraNexthopUpdateIPv6(unittest.TestCase):
    buf = b'\x00\n@ \x01\r\xb8\x00\x00\x00\x00\x00\x00\x00\x14\x01\x01\x00\x00\x00\x02'
    family = socket.AF_INET6
    prefix = '2001:db8::/64'
    metric = 20
    nexthop_num = 1
    nexthop_type = zebra.ZEBRA_NEXTHOP_IFINDEX
    ifindex = 2

    @_patch_frr_v2
    def test_parser(self):
        body = zebra.ZebraNexthopUpdate.parse(self.buf)
        self.assertEqual(self.family, body.family)
        self.assertEqual(self.prefix, body.prefix)
        self.assertEqual(self.metric, body.metric)
        self.assertEqual(self.nexthop_num, len(body.nexthops))
        self.assertEqual(self.nexthop_type, body.nexthops[0].type)
        self.assertEqual(self.ifindex, body.nexthops[0].ifindex)
        buf = body.serialize()
        self.assertEqual(binary_str(self.buf), binary_str(buf))