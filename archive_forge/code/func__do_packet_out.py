import logging
import struct
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import ofp_event
from os_ken.controller.handler import DEAD_DISPATCHER
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
from os_ken.lib import addrconv
from os_ken.lib import hub
from os_ken.lib.dpid import dpid_to_str
from os_ken.lib.packet import packet
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import igmp
def _do_packet_out(self, datapath, data, in_port, actions):
    """send a packet."""
    ofproto = datapath.ofproto
    parser = datapath.ofproto_parser
    out = parser.OFPPacketOut(datapath=datapath, buffer_id=ofproto.OFP_NO_BUFFER, data=data, in_port=in_port, actions=actions)
    datapath.send_msg(out)