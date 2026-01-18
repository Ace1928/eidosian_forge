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
def _generate_config_bpdu(self, flags):
    src_mac = self.ofport.hw_addr
    dst_mac = bpdu.BRIDGE_GROUP_ADDRESS
    length = bpdu.bpdu._PACK_LEN + bpdu.ConfigurationBPDUs.PACK_LEN + llc.llc._PACK_LEN + llc.ControlFormatU._PACK_LEN
    e = ethernet.ethernet(dst_mac, src_mac, length)
    l = llc.llc(llc.SAP_BPDU, llc.SAP_BPDU, llc.ControlFormatU())
    b = bpdu.ConfigurationBPDUs(flags=flags, root_priority=self.port_priority.root_id.priority, root_mac_address=self.port_priority.root_id.mac_addr, root_path_cost=self.port_priority.root_path_cost + self.path_cost, bridge_priority=self.bridge_id.priority, bridge_mac_address=self.bridge_id.mac_addr, port_priority=self.port_id.priority, port_number=self.ofport.port_no, message_age=self.port_times.message_age + 1, max_age=self.port_times.max_age, hello_time=self.port_times.hello_time, forward_delay=self.port_times.forward_delay)
    pkt = packet.Packet()
    pkt.add_protocol(e)
    pkt.add_protocol(l)
    pkt.add_protocol(b)
    pkt.serialize()
    return pkt.data