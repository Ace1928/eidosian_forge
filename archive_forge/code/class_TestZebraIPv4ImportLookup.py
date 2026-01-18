import os
import socket
import sys
import unittest
from unittest import mock
from os_ken.lib import pcaplib
from os_ken.lib.packet import packet
from os_ken.lib.packet import zebra
from os_ken.utils import binary_str
class TestZebraIPv4ImportLookup(unittest.TestCase):
    buf = b'\x18\xc0\xa8\x01\x01'
    prefix = '192.168.1.1/24'
    metric = None
    nexthop_num = 0
    from_zebra = False

    def test_parser(self):
        body = zebra.ZebraIPv4ImportLookup.parse(self.buf)
        self.assertEqual(self.prefix, body.prefix)
        self.assertEqual(self.metric, body.metric)
        self.assertEqual(self.nexthop_num, len(body.nexthops))
        self.assertEqual(self.from_zebra, body.from_zebra)
        buf = body.serialize()
        self.assertEqual(binary_str(self.buf), binary_str(buf))