import logging
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
from os_ken.lib import addrconv
from os_ken.lib.dpid import dpid_to_str
from os_ken.lib.packet import packet
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import slow
def _add_flow_v1_0(self, src, port, timeout, datapath):
    """enter a flow entry for the packet from the slave i/f
        with idle_timeout. for OpenFlow ver1.0."""
    ofproto = datapath.ofproto
    parser = datapath.ofproto_parser
    match = parser.OFPMatch(in_port=port, dl_src=addrconv.mac.text_to_bin(src), dl_type=ether.ETH_TYPE_SLOW)
    actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, 65535)]
    mod = parser.OFPFlowMod(datapath=datapath, match=match, cookie=0, command=ofproto.OFPFC_ADD, idle_timeout=timeout, priority=65535, flags=ofproto.OFPFF_SEND_FLOW_REM, actions=actions)
    datapath.send_msg(mod)