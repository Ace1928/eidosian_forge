import datetime
import logging
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.controller import ofp_event
from os_ken.controller.handler import set_ev_cls
from os_ken.exception import OSKenException
from os_ken.exception import OFPUnknownVersion
from os_ken.lib import hub
from os_ken.lib import mac
from os_ken.lib.dpid import dpid_to_str
from os_ken.lib.packet import bpdu
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import llc
from os_ken.lib.packet import packet
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
class OfCtl_v1_2later(OfCtl_v1_0):

    def __init__(self, dp):
        super(OfCtl_v1_2later, self).__init__(dp)

    def set_port_status(self, port, state):
        ofp = self.dp.ofproto
        parser = self.dp.ofproto_parser
        config = {ofproto_v1_2: PORT_CONFIG_V1_2, ofproto_v1_3: PORT_CONFIG_V1_3}
        mask = 101
        msg = parser.OFPPortMod(self.dp, port.port_no, port.hw_addr, config[ofp][state], mask, port.advertised)
        self.dp.send_msg(msg)
        if config[ofp][state] & ofp.OFPPC_NO_PACKET_IN:
            self.add_no_pkt_in_flow(port.port_no)
        else:
            self.del_no_pkt_in_flow(port.port_no)

    def add_bpdu_pkt_in_flow(self):
        ofp = self.dp.ofproto
        parser = self.dp.ofproto_parser
        match = parser.OFPMatch(eth_dst=bpdu.BRIDGE_GROUP_ADDRESS)
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(self.dp, priority=BPDU_PKT_IN_PRIORITY, match=match, instructions=inst)
        self.dp.send_msg(mod)

    def add_no_pkt_in_flow(self, in_port):
        parser = self.dp.ofproto_parser
        match = parser.OFPMatch(in_port=in_port)
        mod = parser.OFPFlowMod(self.dp, priority=NO_PKT_IN_PRIORITY, match=match)
        self.dp.send_msg(mod)

    def del_no_pkt_in_flow(self, in_port):
        ofp = self.dp.ofproto
        parser = self.dp.ofproto_parser
        match = parser.OFPMatch(in_port=in_port)
        mod = parser.OFPFlowMod(self.dp, command=ofp.OFPFC_DELETE_STRICT, out_port=ofp.OFPP_ANY, out_group=ofp.OFPG_ANY, priority=NO_PKT_IN_PRIORITY, match=match)
        self.dp.send_msg(mod)