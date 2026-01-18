import logging
import array
import netaddr
from os_ken.base import app_manager
from os_ken.controller import dpset
from os_ken.controller import ofp_event
from os_ken.controller import handler
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.lib import mac
from os_ken.lib.packet import packet
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import arp
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import icmp
def _build_ether(self, ethertype, dst_mac=HOST_MAC):
    e = ethernet.ethernet(dst_mac, self.OSKEN_MAC, ethertype)
    return e