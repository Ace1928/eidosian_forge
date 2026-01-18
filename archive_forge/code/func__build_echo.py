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
def _build_echo(self, _type, echo):
    e = self._build_ether(ether.ETH_TYPE_IP)
    ip = ipv4.ipv4(version=4, header_length=5, tos=0, total_length=84, identification=0, flags=0, offset=0, ttl=64, proto=inet.IPPROTO_ICMP, csum=0, src=self.OSKEN_IP, dst=self.HOST_IP)
    ping = icmp.icmp(_type, code=0, csum=0, data=echo)
    p = packet.Packet()
    p.add_protocol(e)
    p.add_protocol(ip)
    p.add_protocol(ping)
    p.serialize()
    return p