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
def _build_arp(self, opcode, dst_ip=HOST_IP):
    if opcode == arp.ARP_REQUEST:
        _eth_dst_mac = self.BROADCAST_MAC
        _arp_dst_mac = self.ZERO_MAC
    elif opcode == arp.ARP_REPLY:
        _eth_dst_mac = self.HOST_MAC
        _arp_dst_mac = self.HOST_MAC
    e = self._build_ether(ether.ETH_TYPE_ARP, _eth_dst_mac)
    a = arp.arp(hwtype=1, proto=ether.ETH_TYPE_IP, hlen=6, plen=4, opcode=opcode, src_mac=self.OSKEN_MAC, src_ip=self.OSKEN_IP, dst_mac=_arp_dst_mac, dst_ip=dst_ip)
    p = packet.Packet()
    p.add_protocol(e)
    p.add_protocol(a)
    p.serialize()
    return p