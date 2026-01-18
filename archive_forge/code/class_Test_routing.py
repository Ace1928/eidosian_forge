import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib.packet import ipv6
class Test_routing(unittest.TestCase):

    def setUp(self):
        self.nxt = 0
        self.size = 6
        self.type_ = ipv6.routing.ROUTING_TYPE_3
        self.seg = 0
        self.cmpi = 0
        self.cmpe = 0
        self.adrs = ['2001:db8:dead::1', '2001:db8:dead::2', '2001:db8:dead::3']
        self.pad = (8 - ((len(self.adrs) - 1) * (16 - self.cmpi) + (16 - self.cmpe) % 8)) % 8
        self.form = '!BBBBBB2x16s16s16s'
        self.buf = struct.pack(self.form, self.nxt, self.size, self.type_, self.seg, self.cmpi << 4 | self.cmpe, self.pad << 4, addrconv.ipv6.text_to_bin(self.adrs[0]), addrconv.ipv6.text_to_bin(self.adrs[1]), addrconv.ipv6.text_to_bin(self.adrs[2]))

    def tearDown(self):
        pass

    def test_parser(self):
        _res = ipv6.routing.parser(self.buf)
        if type(_res) is tuple:
            res = _res[0]
        else:
            res = _res
        self.assertEqual(self.nxt, res.nxt)
        self.assertEqual(self.size, res.size)
        self.assertEqual(self.type_, res.type_)
        self.assertEqual(self.seg, res.seg)
        self.assertEqual(self.cmpi, res.cmpi)
        self.assertEqual(self.cmpe, res.cmpe)
        self.assertEqual(self.pad, res._pad)
        self.assertEqual(self.adrs[0], res.adrs[0])
        self.assertEqual(self.adrs[1], res.adrs[1])
        self.assertEqual(self.adrs[2], res.adrs[2])

    def test_not_implemented_type(self):
        not_implemented_buf = struct.pack('!BBBBBB2x', 0, 6, ipv6.routing.ROUTING_TYPE_2, 0, 0, 0)
        instance = ipv6.routing.parser(not_implemented_buf)
        assert None is instance

    def test_invalid_type(self):
        invalid_type = 99
        invalid_buf = struct.pack('!BBBBBB2x', 0, 6, invalid_type, 0, 0, 0)
        instance = ipv6.routing.parser(invalid_buf)
        assert None is instance