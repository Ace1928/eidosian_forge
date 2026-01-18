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
def _build_mldv2_query(self):
    e = ethernet(ethertype=ether.ETH_TYPE_IPV6)
    i = ipv6(nxt=inet.IPPROTO_ICMPV6)
    ic = icmpv6.icmpv6(type_=icmpv6.MLD_LISTENER_QUERY, data=self.mld)
    p = e / i / ic
    return p