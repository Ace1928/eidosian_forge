import unittest
import logging
import struct
import inspect
from os_ken.ofproto import ether, inet
from os_ken.lib.packet.ethernet import ethernet
from os_ken.lib.packet import icmpv6
from os_ken.lib.packet.ipv6 import ipv6
from os_ken.lib.packet import packet_utils
from os_ken.lib import addrconv
class Test_icmpv6_nd_option_la(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_default_args(self):
        la = icmpv6.nd_option_sla()
        buf = la.serialize()
        res = struct.unpack(icmpv6.nd_option_sla._PACK_STR, bytes(buf))
        self.assertEqual(res[0], icmpv6.ND_OPTION_SLA)
        self.assertEqual(res[1], len(icmpv6.nd_option_sla()) // 8)
        self.assertEqual(res[2], addrconv.mac.text_to_bin('00:00:00:00:00:00'))
        prev = ipv6(nxt=inet.IPPROTO_ICMPV6)
        ic = icmpv6.icmpv6(type_=icmpv6.ND_NEIGHBOR_ADVERT, data=icmpv6.nd_neighbor(option=icmpv6.nd_option_tla()))
        prev.serialize(ic, None)
        buf = ic.serialize(bytearray(), prev)
        res = struct.unpack(icmpv6.icmpv6._PACK_STR, bytes(buf[:4]))
        self.assertEqual(res[0], icmpv6.ND_NEIGHBOR_ADVERT)
        self.assertEqual(res[1], 0)
        self.assertEqual(res[2], icmpv6_csum(prev, buf))
        res = struct.unpack(icmpv6.nd_neighbor._PACK_STR, bytes(buf[4:24]))
        self.assertEqual(res[0], 0)
        self.assertEqual(res[1], addrconv.ipv6.text_to_bin('::'))
        res = struct.unpack(icmpv6.nd_option_tla._PACK_STR, bytes(buf[24:]))
        self.assertEqual(res[0], icmpv6.ND_OPTION_TLA)
        self.assertEqual(res[1], len(icmpv6.nd_option_tla()) // 8)
        self.assertEqual(res[2], addrconv.mac.text_to_bin('00:00:00:00:00:00'))
        prev = ipv6(nxt=inet.IPPROTO_ICMPV6)
        ic = icmpv6.icmpv6(type_=icmpv6.ND_ROUTER_SOLICIT, data=icmpv6.nd_router_solicit(option=icmpv6.nd_option_sla()))
        prev.serialize(ic, None)
        buf = ic.serialize(bytearray(), prev)
        res = struct.unpack(icmpv6.icmpv6._PACK_STR, bytes(buf[:4]))
        self.assertEqual(res[0], icmpv6.ND_ROUTER_SOLICIT)
        self.assertEqual(res[1], 0)
        self.assertEqual(res[2], icmpv6_csum(prev, buf))
        res = struct.unpack(icmpv6.nd_router_solicit._PACK_STR, bytes(buf[4:8]))
        self.assertEqual(res[0], 0)
        res = struct.unpack(icmpv6.nd_option_sla._PACK_STR, bytes(buf[8:]))
        self.assertEqual(res[0], icmpv6.ND_OPTION_SLA)
        self.assertEqual(res[1], len(icmpv6.nd_option_sla()) // 8)
        self.assertEqual(res[2], addrconv.mac.text_to_bin('00:00:00:00:00:00'))