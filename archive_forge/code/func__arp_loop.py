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
def _arp_loop(self):
    try:
        with contextlib.closing(socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(ether.ETH_TYPE_ARP))) as packet_socket:
            packet_socket.bind((self.interface.device_name, socket.htons(ether.ETH_TYPE_ARP), socket.PACKET_BROADCAST, arp.ARP_HW_TYPE_ETHERNET, mac_lib.BROADCAST))
            self._arp_loop_socket(packet_socket)
    except greenlet.GreenletExit:
        pass