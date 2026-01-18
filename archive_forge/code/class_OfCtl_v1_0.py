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
class OfCtl_v1_0(object):

    def __init__(self, dp):
        super(OfCtl_v1_0, self).__init__()
        self.dp = dp

    def send_packet_out(self, out_port, data):
        actions = [self.dp.ofproto_parser.OFPActionOutput(out_port, 0)]
        self.dp.send_packet_out(buffer_id=self.dp.ofproto.OFP_NO_BUFFER, in_port=self.dp.ofproto.OFPP_CONTROLLER, actions=actions, data=data)

    def set_port_status(self, port, state):
        ofproto_parser = self.dp.ofproto_parser
        mask = 127
        msg = ofproto_parser.OFPPortMod(self.dp, port.port_no, port.hw_addr, PORT_CONFIG_V1_0[state], mask, port.advertised)
        self.dp.send_msg(msg)