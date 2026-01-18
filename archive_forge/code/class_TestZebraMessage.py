import os
import socket
import sys
import unittest
from unittest import mock
from os_ken.lib import pcaplib
from os_ken.lib.packet import packet
from os_ken.lib.packet import zebra
from os_ken.utils import binary_str
class TestZebraMessage(unittest.TestCase):

    def test_get_header_size(self):
        self.assertEqual(zebra.ZebraMessage.V0_HEADER_SIZE, zebra.ZebraMessage.get_header_size(0))
        self.assertEqual(zebra.ZebraMessage.V1_HEADER_SIZE, zebra.ZebraMessage.get_header_size(2))
        self.assertEqual(zebra.ZebraMessage.V3_HEADER_SIZE, zebra.ZebraMessage.get_header_size(3))
        self.assertEqual(zebra.ZebraMessage.V3_HEADER_SIZE, zebra.ZebraMessage.get_header_size(4))

    def test_get_header_size_invalid_version(self):
        self.assertRaises(ValueError, zebra.ZebraMessage.get_header_size, 255)