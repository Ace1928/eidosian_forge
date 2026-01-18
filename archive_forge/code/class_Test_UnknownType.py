import unittest
import logging
import struct
from os_ken.lib.packet import bpdu
class Test_UnknownType(unittest.TestCase):
    """ Test case for unknown BPDU type
    """

    def setUp(self):
        self.protocol_id = bpdu.PROTOCOL_IDENTIFIER
        self.version_id = bpdu.RstBPDUs.VERSION_ID
        self.bpdu_type = 222
        self.flags = 126
        self.root_priority = 4096
        self.root_system_id_extension = 1
        self.root_mac_address = '12:34:56:78:9a:bc'
        self.root_path_cost = 2
        self.bridge_priority = 8192
        self.bridge_system_id_extension = 3
        self.bridge_mac_address = 'aa:aa:aa:aa:aa:aa'
        self.port_priority = 16
        self.port_number = 4
        self.message_age = 5
        self.max_age = 6
        self.hello_time = 7
        self.forward_delay = 8
        self.version_1_length = bpdu.VERSION_1_LENGTH
        self.fmt = bpdu.bpdu._PACK_STR + bpdu.ConfigurationBPDUs._PACK_STR[1:] + bpdu.RstBPDUs._PACK_STR[1:]
        self.buf = struct.pack(self.fmt, self.protocol_id, self.version_id, self.bpdu_type, self.flags, bpdu.RstBPDUs.encode_bridge_id(self.root_priority, self.root_system_id_extension, self.root_mac_address), self.root_path_cost, bpdu.RstBPDUs.encode_bridge_id(self.bridge_priority, self.bridge_system_id_extension, self.bridge_mac_address), bpdu.RstBPDUs.encode_port_id(self.port_priority, self.port_number), bpdu.RstBPDUs._encode_timer(self.message_age), bpdu.RstBPDUs._encode_timer(self.max_age), bpdu.RstBPDUs._encode_timer(self.hello_time), bpdu.RstBPDUs._encode_timer(self.forward_delay), self.version_1_length)

    def test_parser(self):
        r1, r2, _ = bpdu.bpdu.parser(self.buf)
        self.assertEqual(r1, self.buf)
        self.assertEqual(r2, None)