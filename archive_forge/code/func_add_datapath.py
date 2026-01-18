import collections
from os_ken.base import app_manager
import os_ken.exception as os_ken_exc
from os_ken.controller import event
from os_ken.exception import NetworkNotFound, NetworkAlreadyExist
from os_ken.exception import PortAlreadyExist, PortNotFound, PortUnknown
def add_datapath(self, ofp_switch_features):
    datapath = ofp_switch_features.datapath
    dpid = ofp_switch_features.datapath_id
    ports = ofp_switch_features.ports
    self.dpids.setdefault_dpid(dpid)
    for port_no in ports:
        self.port_added(datapath, port_no)