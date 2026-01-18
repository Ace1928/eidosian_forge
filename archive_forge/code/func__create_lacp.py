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
def _create_lacp(self, datapath, port, req):
    """create a LACP packet."""
    actor_system = datapath.ports[datapath.ofproto.OFPP_LOCAL].hw_addr
    res = slow.lacp(actor_system_priority=65535, actor_system=actor_system, actor_key=req.actor_key, actor_port_priority=255, actor_port=port, actor_state_activity=req.LACP_STATE_PASSIVE, actor_state_timeout=req.actor_state_timeout, actor_state_aggregation=req.actor_state_aggregation, actor_state_synchronization=req.actor_state_synchronization, actor_state_collecting=req.actor_state_collecting, actor_state_distributing=req.actor_state_distributing, actor_state_defaulted=req.LACP_STATE_OPERATIONAL_PARTNER, actor_state_expired=req.LACP_STATE_NOT_EXPIRED, partner_system_priority=req.actor_system_priority, partner_system=req.actor_system, partner_key=req.actor_key, partner_port_priority=req.actor_port_priority, partner_port=req.actor_port, partner_state_activity=req.actor_state_activity, partner_state_timeout=req.actor_state_timeout, partner_state_aggregation=req.actor_state_aggregation, partner_state_synchronization=req.actor_state_synchronization, partner_state_collecting=req.actor_state_collecting, partner_state_distributing=req.actor_state_distributing, partner_state_defaulted=req.actor_state_defaulted, partner_state_expired=req.actor_state_expired, collector_max_delay=0)
    self.logger.info('SW=%s PORT=%d LACP sent.', dpid_to_str(datapath.id), port)
    self.logger.debug(str(res))
    return res