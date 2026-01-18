import contextlib
import socket
import struct
from os_ken.controller import handler
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.lib import addrconv
from os_ken.lib import hub
from os_ken.lib.packet import arp
from os_ken.lib.packet import vrrp
from os_ken.services.protocols.vrrp import monitor
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import utils
def _join_multicast_membership(self, join_leave):
    config = self.config
    if config.is_ipv6:
        mac_address = vrrp.vrrp_ipv6_src_mac_address(config.vrid)
    else:
        mac_address = vrrp.vrrp_ipv4_src_mac_address(config.vrid)
    if join_leave:
        add_drop = PACKET_ADD_MEMBERSHIP
    else:
        add_drop = PACKET_DROP_MEMBERSHIP
    packet_mreq = struct.pack('IHH8s', self.ifindex, PACKET_MR_MULTICAST, 6, addrconv.mac.text_to_bin(mac_address))
    self.packet_socket.setsockopt(SOL_PACKET, add_drop, packet_mreq)