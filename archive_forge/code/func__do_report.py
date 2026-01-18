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
def _do_report(self, report, in_port, msg):
    """the process when the snooper received a REPORT message."""
    datapath = msg.datapath
    dpid = datapath.id
    ofproto = datapath.ofproto
    parser = datapath.ofproto_parser
    if ofproto.OFP_VERSION == ofproto_v1_0.OFP_VERSION:
        size = 65535
    else:
        size = ofproto.OFPCML_MAX
    outport = None
    value = self._to_querier.get(dpid)
    if value:
        outport = value['port']
    self._to_hosts.setdefault(dpid, {})
    if not self._to_hosts[dpid].get(report.address):
        self._send_event(EventMulticastGroupStateChanged(MG_GROUP_ADDED, report.address, outport, []))
        self._to_hosts[dpid].setdefault(report.address, {'replied': False, 'leave': None, 'ports': {}})
    if not self._to_hosts[dpid][report.address]['ports'].get(in_port):
        self._to_hosts[dpid][report.address]['ports'][in_port] = {'out': False, 'in': False}
        self._set_flow_entry(datapath, [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, size)], in_port, report.address)
    if not self._to_hosts[dpid][report.address]['ports'][in_port]['out']:
        self._to_hosts[dpid][report.address]['ports'][in_port]['out'] = True
    if not outport:
        self.logger.info('no querier exists.')
        return
    if not self._to_hosts[dpid][report.address]['ports'][in_port]['in']:
        actions = []
        ports = []
        for port in self._to_hosts[dpid][report.address]['ports']:
            actions.append(parser.OFPActionOutput(port))
            ports.append(port)
        self._send_event(EventMulticastGroupStateChanged(MG_MEMBER_CHANGED, report.address, outport, ports))
        self._set_flow_entry(datapath, actions, outport, report.address)
        self._to_hosts[dpid][report.address]['ports'][in_port]['in'] = True
    if not self._to_hosts[dpid][report.address]['replied']:
        actions = [parser.OFPActionOutput(outport, size)]
        self._do_packet_out(datapath, msg.data, in_port, actions)
        self._to_hosts[dpid][report.address]['replied'] = True