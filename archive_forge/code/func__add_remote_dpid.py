import collections
import logging
import os_ken.exception as os_ken_exc
from os_ken.base import app_manager
from os_ken.controller import event
def _add_remote_dpid(self, dpid, port_no, remote_dpid):
    self.dpids[dpid][port_no] = remote_dpid
    self.send_event(EventTunnelPort(dpid, port_no, remote_dpid, True))