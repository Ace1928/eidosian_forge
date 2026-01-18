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
def _select_designated_port(self, root_port):
    """ DESIGNATED_PORT is a port of the side near the root bridge
            of each link. It is determined by the cost of each path, etc
            same as ROOT_PORT. """
    d_ports = []
    root_msg = root_port.designated_priority
    for port in self.ports.values():
        port_msg = port.designated_priority
        if port.state is PORT_STATE_DISABLE or port.ofport.port_no == root_port.ofport.port_no:
            continue
        if port_msg is None or port_msg.root_id.value != root_msg.root_id.value:
            d_ports.append(port.ofport.port_no)
        else:
            result = Stp.compare_root_path(root_msg.root_path_cost, port_msg.root_path_cost - port.path_cost, self.bridge_id.value, port_msg.designated_bridge_id.value, port.port_id.value, port_msg.designated_port_id.value)
            if result is SUPERIOR:
                d_ports.append(port.ofport.port_no)
    return d_ports