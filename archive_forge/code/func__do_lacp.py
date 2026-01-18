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
def _do_lacp(self, req_lacp, src, msg):
    """packet-in process when the received packet is LACP."""
    datapath = msg.datapath
    dpid = datapath.id
    ofproto = datapath.ofproto
    parser = datapath.ofproto_parser
    if ofproto.OFP_VERSION == ofproto_v1_0.OFP_VERSION:
        port = msg.in_port
    else:
        port = msg.match['in_port']
    self.logger.info('SW=%s PORT=%d LACP received.', dpid_to_str(dpid), port)
    self.logger.debug(str(req_lacp))
    if not self._get_slave_enabled(dpid, port):
        self.logger.info('SW=%s PORT=%d the slave i/f has just been up.', dpid_to_str(dpid), port)
        self._set_slave_enabled(dpid, port, True)
        self.send_event_to_observers(EventSlaveStateChanged(datapath, port, True))
    if req_lacp.LACP_STATE_SHORT_TIMEOUT == req_lacp.actor_state_timeout:
        idle_timeout = req_lacp.SHORT_TIMEOUT_TIME
    else:
        idle_timeout = req_lacp.LONG_TIMEOUT_TIME
    if idle_timeout != self._get_slave_timeout(dpid, port):
        self.logger.info('SW=%s PORT=%d the timeout time has changed.', dpid_to_str(dpid), port)
        self._set_slave_timeout(dpid, port, idle_timeout)
        func = self._add_flow.get(ofproto.OFP_VERSION)
        assert func
        func(src, port, idle_timeout, datapath)
    res_pkt = self._create_response(datapath, port, req_lacp)
    out_port = ofproto.OFPP_IN_PORT
    actions = [parser.OFPActionOutput(out_port)]
    out = datapath.ofproto_parser.OFPPacketOut(datapath=datapath, buffer_id=ofproto.OFP_NO_BUFFER, data=res_pkt.data, in_port=port, actions=actions)
    datapath.send_msg(out)