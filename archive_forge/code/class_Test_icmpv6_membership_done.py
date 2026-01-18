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
class Test_icmpv6_membership_done(Test_icmpv6_membership_query):
    type_ = 132
    code = 0
    csum = 45988
    maxresp = 10000
    address = 'ff08::1'
    buf = b"\x84\x00\xb3\xa4'\x10\x00\x00" + b'\xff\x08\x00\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00\x00\x00\x00\x01'

    def test_json(self):
        ic1 = icmpv6.icmpv6(type_=icmpv6.MLD_LISTENER_DONE, data=icmpv6.mld())
        jsondict = ic1.to_jsondict()
        ic2 = icmpv6.icmpv6.from_jsondict(jsondict['icmpv6'])
        self.assertEqual(str(ic1), str(ic2))