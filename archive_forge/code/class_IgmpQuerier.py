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
class IgmpQuerier(IgmpBase):
    """IGMP querier emulation class library.

    this querier is a simplified implementation, and is not based on RFC,
    for example as following points:
    - ignore some constant values
    - does not send a specific QUERY in response to LEAVE
    - and so on
    """

    def __init__(self):
        """initialization."""
        super(IgmpQuerier, self).__init__()
        self.name = 'IgmpQuerier'
        self.logger = logging.getLogger(self.name)
        self.dpid = None
        self.server_port = None
        self._datapath = None
        self._querier_thread = None
        self._mcast = {}
        self._set_logger()

    def set_querier_mode(self, dpid, server_port):
        """set the datapath to work as a querier. note that you can set
        up only the one querier. when you called this method several
        times, only the last one becomes effective."""
        self.dpid = dpid
        self.server_port = server_port
        if self._querier_thread:
            hub.kill(self._querier_thread)
            self._querier_thread = None

    def packet_in_handler(self, req_igmp, msg):
        """the process when the querier received IGMP."""
        ofproto = msg.datapath.ofproto
        if ofproto.OFP_VERSION == ofproto_v1_0.OFP_VERSION:
            in_port = msg.in_port
        else:
            in_port = msg.match['in_port']
        if igmp.IGMP_TYPE_REPORT_V1 == req_igmp.msgtype or igmp.IGMP_TYPE_REPORT_V2 == req_igmp.msgtype:
            self._do_report(req_igmp, in_port, msg)
        elif igmp.IGMP_TYPE_LEAVE == req_igmp.msgtype:
            self._do_leave(req_igmp, in_port, msg)

    def start_loop(self, datapath):
        """start QUERY thread."""
        self._datapath = datapath
        self._querier_thread = hub.spawn(self._send_query)
        self.logger.info('started a querier.')

    def stop_loop(self):
        """stop QUERY thread."""
        hub.kill(self._querier_thread)
        self._querier_thread = None
        self._datapath = None
        self.logger.info('stopped a querier.')

    def _send_query(self):
        """ send a QUERY message periodically."""
        timeout = 60
        ofproto = self._datapath.ofproto
        parser = self._datapath.ofproto_parser
        if ofproto_v1_0.OFP_VERSION == ofproto.OFP_VERSION:
            send_port = ofproto.OFPP_NONE
        else:
            send_port = ofproto.OFPP_ANY
        res_igmp = igmp.igmp(msgtype=igmp.IGMP_TYPE_QUERY, maxresp=igmp.QUERY_RESPONSE_INTERVAL * 10, csum=0, address='0.0.0.0')
        res_ipv4 = ipv4.ipv4(total_length=len(ipv4.ipv4()) + len(res_igmp), proto=inet.IPPROTO_IGMP, ttl=1, src='0.0.0.0', dst=igmp.MULTICAST_IP_ALL_HOST)
        res_ether = ethernet.ethernet(dst=igmp.MULTICAST_MAC_ALL_HOST, src=self._datapath.ports[ofproto.OFPP_LOCAL].hw_addr, ethertype=ether.ETH_TYPE_IP)
        res_pkt = packet.Packet()
        res_pkt.add_protocol(res_ether)
        res_pkt.add_protocol(res_ipv4)
        res_pkt.add_protocol(res_igmp)
        res_pkt.serialize()
        flood = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
        while True:
            for status in self._mcast.values():
                for port in status.keys():
                    status[port] = False
            self._do_packet_out(self._datapath, res_pkt.data, send_port, flood)
            hub.sleep(igmp.QUERY_RESPONSE_INTERVAL)
            del_groups = []
            for group, status in self._mcast.items():
                del_ports = []
                actions = []
                for port in status.keys():
                    if not status[port]:
                        del_ports.append(port)
                    else:
                        actions.append(parser.OFPActionOutput(port))
                if len(actions) and len(del_ports):
                    self._set_flow_entry(self._datapath, actions, self.server_port, group)
                if not len(actions):
                    self._del_flow_entry(self._datapath, self.server_port, group)
                    del_groups.append(group)
                if len(del_ports):
                    for port in del_ports:
                        self._del_flow_entry(self._datapath, port, group)
                for port in del_ports:
                    del status[port]
            for group in del_groups:
                del self._mcast[group]
            rest_time = timeout - igmp.QUERY_RESPONSE_INTERVAL
            hub.sleep(rest_time)

    def _do_report(self, report, in_port, msg):
        """the process when the querier received a REPORT message."""
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        if ofproto.OFP_VERSION == ofproto_v1_0.OFP_VERSION:
            size = 65535
        else:
            size = ofproto.OFPCML_MAX
        update = False
        self._mcast.setdefault(report.address, {})
        if in_port not in self._mcast[report.address]:
            update = True
        self._mcast[report.address][in_port] = True
        if update:
            actions = []
            for port in self._mcast[report.address]:
                actions.append(parser.OFPActionOutput(port))
            self._set_flow_entry(datapath, actions, self.server_port, report.address)
            self._set_flow_entry(datapath, [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, size)], in_port, report.address)

    def _do_leave(self, leave, in_port, msg):
        """the process when the querier received a LEAVE message."""
        datapath = msg.datapath
        parser = datapath.ofproto_parser
        self._mcast.setdefault(leave.address, {})
        if in_port in self._mcast[leave.address]:
            self._del_flow_entry(datapath, in_port, leave.address)
            del self._mcast[leave.address][in_port]
            actions = []
            for port in self._mcast[leave.address]:
                actions.append(parser.OFPActionOutput(port))
            if len(actions):
                self._set_flow_entry(datapath, actions, self.server_port, leave.address)
            else:
                self._del_flow_entry(datapath, self.server_port, leave.address)

    def _set_logger(self):
        """change log format."""
        self.logger.propagate = False
        hdl = logging.StreamHandler()
        fmt_str = '[querier][%(levelname)s] %(message)s'
        hdl.setFormatter(logging.Formatter(fmt_str))
        self.logger.addHandler(hdl)