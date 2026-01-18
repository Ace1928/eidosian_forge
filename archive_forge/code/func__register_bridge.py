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
def _register_bridge(self, dp):
    self._unregister_bridge(dp.id)
    dpid_str = {'dpid': dpid_to_str(dp.id)}
    self.logger.info('Join as stp bridge.', extra=dpid_str)
    try:
        bridge = Bridge(dp, self.logger, self.config.get(dp.id, {}), self.send_event_to_observers)
    except OFPUnknownVersion as message:
        self.logger.error(str(message), extra=dpid_str)
        return
    self.bridge_list[dp.id] = bridge