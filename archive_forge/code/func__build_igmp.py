import unittest
import inspect
import logging
from struct import pack, unpack_from, pack_into
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.lib.packet.ethernet import ethernet
from os_ken.lib.packet.ipv4 import ipv4
from os_ken.lib.packet.packet import Packet
from os_ken.lib.packet.packet_utils import checksum
from os_ken.lib import addrconv
from os_ken.lib.packet.igmp import igmp
from os_ken.lib.packet.igmp import igmpv3_query
from os_ken.lib.packet.igmp import igmpv3_report
from os_ken.lib.packet.igmp import igmpv3_report_group
from os_ken.lib.packet.igmp import IGMP_TYPE_QUERY
from os_ken.lib.packet.igmp import IGMP_TYPE_REPORT_V3
from os_ken.lib.packet.igmp import MODE_IS_INCLUDE
def _build_igmp(self):
    dl_dst = '11:22:33:44:55:66'
    dl_src = 'aa:bb:cc:dd:ee:ff'
    dl_type = ether.ETH_TYPE_IP
    e = ethernet(dl_dst, dl_src, dl_type)
    total_length = len(ipv4()) + len(self.g)
    nw_proto = inet.IPPROTO_IGMP
    nw_dst = '11.22.33.44'
    nw_src = '55.66.77.88'
    i = ipv4(total_length=total_length, src=nw_src, dst=nw_dst, proto=nw_proto, ttl=1)
    p = Packet()
    p.add_protocol(e)
    p.add_protocol(i)
    p.add_protocol(self.g)
    p.serialize()
    return p