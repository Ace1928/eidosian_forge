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
def _send_msg(self, dp, data):
    buffer_id = 4294967295
    in_port = dp.ofproto.OFPP_LOCAL
    actions = [dp.ofproto_parser.OFPActionOutput(1, 0)]
    msg = dp.ofproto_parser.OFPPacketOut(dp, buffer_id, in_port, actions, data)
    dp.send_msg(msg)