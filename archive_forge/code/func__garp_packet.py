import contextlib
import greenlet
import socket
from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.controller import ofp_event
from os_ken.lib import hub
from os_ken.lib import mac as mac_lib
from os_ken.lib.packet import arp
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import packet
from os_ken.lib.packet import vlan
from os_ken.lib.packet import vrrp
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_v1_2
from os_ken.services.protocols.vrrp import api as vrrp_api
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import utils
def _garp_packet(self, ip_address):
    src_mac = vrrp.vrrp_ipv4_src_mac_address(self.config.vrid)
    e = ethernet.ethernet(mac_lib.BROADCAST_STR, src_mac, ether.ETH_TYPE_ARP)
    a = arp.arp_ip(arp.ARP_REQUEST, src_mac, ip_address, mac_lib.DONTCARE_STR, ip_address)
    p = packet.Packet()
    p.add_protocol(e)
    utils.may_add_vlan(p, self.interface.vlan_id)
    p.add_protocol(a)
    p.serialize()
    return p