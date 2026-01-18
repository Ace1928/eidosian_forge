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
def _spanning_tree_algorithm(self):
    """ Update tree roles.
             - Root bridge:
                all port is DESIGNATED_PORT.
             - Non root bridge:
                select one ROOT_PORT and some DESIGNATED_PORT,
                and the other port is set to NON_DESIGNATED_PORT."""
    port_roles = {}
    root_port = self._select_root_port()
    if root_port is None:
        self.logger.info('Root bridge.', extra=self.dpid_str)
        root_priority = self.root_priority
        root_times = self.root_times
        for port_no in self.ports:
            if self.ports[port_no].state is not PORT_STATE_DISABLE:
                port_roles[port_no] = DESIGNATED_PORT
    else:
        self.logger.info('Non root bridge.', extra=self.dpid_str)
        root_priority = root_port.designated_priority
        root_times = root_port.designated_times
        port_roles[root_port.ofport.port_no] = ROOT_PORT
        d_ports = self._select_designated_port(root_port)
        for port_no in d_ports:
            port_roles[port_no] = DESIGNATED_PORT
        for port in self.ports.values():
            if port.state is not PORT_STATE_DISABLE:
                port_roles.setdefault(port.ofport.port_no, NON_DESIGNATED_PORT)
    return (port_roles, root_priority, root_times)