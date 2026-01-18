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
def _arp_match(self, dp):
    kwargs = {}
    kwargs['in_port'] = self.interface.port_no
    kwargs['eth_dst'] = mac_lib.BROADCAST_STR
    kwargs['eth_type'] = ether.ETH_TYPE_ARP
    if self.interface.vlan_id is not None:
        kwargs['vlan_vid'] = self.interface.vlan_id
    kwargs['arp_op'] = arp.ARP_REQUEST
    kwargs['arp_tpa'] = vrrp.vrrp_ipv4_src_mac_address(self.config.vrid)
    return dp.ofproto_parser.OFPMatch(**kwargs)