import os
import socket
import sys
import unittest
from unittest import mock
from os_ken.lib import pcaplib
from os_ken.lib.packet import packet
from os_ken.lib.packet import zebra
from os_ken.utils import binary_str
class TestZebraInterfaceVrfUpdate(unittest.TestCase):
    buf = b'\x00\x00\x00\x01\x00\x02'
    ifindex = 1
    vrf_id = 2

    @_patch_frr_v2
    def test_parser(self):
        body = zebra.ZebraInterfaceVrfUpdate.parse(self.buf)
        self.assertEqual(self.ifindex, body.ifindex)
        self.assertEqual(self.vrf_id, body.vrf_id)
        buf = body.serialize()
        self.assertEqual(binary_str(self.buf), binary_str(buf))