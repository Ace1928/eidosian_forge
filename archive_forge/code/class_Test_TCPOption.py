import unittest
import logging
import struct
from struct import *
from os_ken.ofproto import inet
from os_ken.lib.packet import tcp
from os_ken.lib.packet.ipv4 import ipv4
from os_ken.lib.packet import packet_utils
from os_ken.lib import addrconv
class Test_TCPOption(unittest.TestCase):
    input_options = [tcp.TCPOptionEndOfOptionList(), tcp.TCPOptionNoOperation(), tcp.TCPOptionMaximumSegmentSize(max_seg_size=1460), tcp.TCPOptionWindowScale(shift_cnt=9), tcp.TCPOptionSACKPermitted(), tcp.TCPOptionSACK(blocks=[(1, 2), (3, 4)], length=18), tcp.TCPOptionTimestamps(ts_val=287454020, ts_ecr=1432778632), tcp.TCPOptionUserTimeout(granularity=1, user_timeout=564), tcp.TCPOptionAuthentication(key_id=1, r_next_key_id=2, mac=b'abcdefghijkl', length=16), tcp.TCPOptionUnknown(value=b'foobar', kind=255, length=8), tcp.TCPOptionUnknown(value=b'', kind=255, length=2)]
    input_buf = b'\x00\x01\x02\x04\x05\xb4\x03\x03\t\x04\x02\x05\x12\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x08\n\x11"3DUfw\x88\x1c\x04\x824\x1d\x10\x01\x02abcdefghijkl\xff\x08foobar\xff\x02'

    def test_serialize(self):
        output_buf = bytearray()
        for option in self.input_options:
            output_buf += option.serialize()
        self.assertEqual(self.input_buf, output_buf)

    def test_parser(self):
        buf = self.input_buf
        output_options = []
        while buf:
            opt, buf = tcp.TCPOption.parser(buf)
            output_options.append(opt)
        self.assertEqual(str(self.input_options), str(output_options))

    def test_json(self):
        for option in self.input_options:
            json_dict = option.to_jsondict()[option.__class__.__name__]
            output_option = option.__class__.from_jsondict(json_dict)
            self.assertEqual(str(option), str(output_option))