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
def _do_timeout_for_leave(self, timeout, datapath, dst, in_port):
    """the process when the QUERY from the switch timeout expired."""
    parser = datapath.ofproto_parser
    dpid = datapath.id
    hub.sleep(timeout)
    outport = self._to_querier[dpid]['port']
    if self._to_hosts[dpid][dst]['ports'][in_port]['out']:
        return
    del self._to_hosts[dpid][dst]['ports'][in_port]
    self._del_flow_entry(datapath, in_port, dst)
    actions = []
    ports = []
    for port in self._to_hosts[dpid][dst]['ports']:
        actions.append(parser.OFPActionOutput(port))
        ports.append(port)
    if len(actions):
        self._send_event(EventMulticastGroupStateChanged(MG_MEMBER_CHANGED, dst, outport, ports))
        self._set_flow_entry(datapath, actions, outport, dst)
        self._to_hosts[dpid][dst]['leave'] = None
    else:
        self._remove_multicast_group(datapath, outport, dst)
        del self._to_hosts[dpid][dst]