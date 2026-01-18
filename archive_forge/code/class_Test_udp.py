import unittest
import logging
import struct
from struct import *
from os_ken.ofproto import ether, inet
from os_ken.lib.packet.packet import Packet
from os_ken.lib.packet.udp import udp
from os_ken.lib.packet.ipv4 import ipv4
from os_ken.lib.packet import packet_utils
from os_ken.lib import addrconv
class Test_udp(unittest.TestCase):
    """ Test case for udp
    """
    src_port = 6431
    dst_port = 8080
    total_length = 65507
    csum = 12345
    u = udp(src_port, dst_port, total_length, csum)
    buf = pack(udp._PACK_STR, src_port, dst_port, total_length, csum)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.src_port, self.u.src_port)
        self.assertEqual(self.dst_port, self.u.dst_port)
        self.assertEqual(self.total_length, self.u.total_length)
        self.assertEqual(self.csum, self.u.csum)

    def test_parser(self):
        r1, r2, _ = self.u.parser(self.buf)
        self.assertEqual(self.src_port, r1.src_port)
        self.assertEqual(self.dst_port, r1.dst_port)
        self.assertEqual(self.total_length, r1.total_length)
        self.assertEqual(self.csum, r1.csum)
        self.assertEqual(None, r2)

    def test_serialize(self):
        src_port = 6431
        dst_port = 8080
        total_length = 0
        csum = 0
        src_ip = '192.168.10.1'
        dst_ip = '192.168.100.1'
        prev = ipv4(4, 5, 0, 0, 0, 0, 0, 64, inet.IPPROTO_UDP, 0, src_ip, dst_ip)
        u = udp(src_port, dst_port, total_length, csum)
        buf = u.serialize(bytearray(), prev)
        res = struct.unpack(udp._PACK_STR, buf)
        self.assertEqual(res[0], src_port)
        self.assertEqual(res[1], dst_port)
        self.assertEqual(res[2], struct.calcsize(udp._PACK_STR))
        ph = struct.pack('!4s4sBBH', addrconv.ipv4.text_to_bin(src_ip), addrconv.ipv4.text_to_bin(dst_ip), 0, 17, res[2])
        d = ph + buf + bytearray()
        s = packet_utils.checksum(d)
        self.assertEqual(0, s)

    def test_malformed_udp(self):
        m_short_buf = self.buf[1:udp._MIN_LEN]
        self.assertRaises(Exception, udp.parser, m_short_buf)

    def test_default_args(self):
        prev = ipv4(proto=inet.IPPROTO_UDP)
        u = udp()
        buf = u.serialize(bytearray(), prev)
        res = struct.unpack(udp._PACK_STR, buf)
        self.assertEqual(res[0], 1)
        self.assertEqual(res[1], 1)
        self.assertEqual(res[2], udp._MIN_LEN)

    def test_json(self):
        jsondict = self.u.to_jsondict()
        u = udp.from_jsondict(jsondict['udp'])
        self.assertEqual(str(self.u), str(u))