import collections
from os_ken.base import app_manager
import os_ken.exception as os_ken_exc
from os_ken.controller import event
from os_ken.exception import NetworkNotFound, NetworkAlreadyExist
from os_ken.exception import PortAlreadyExist, PortNotFound, PortUnknown
def _setdefault_network(self, dpid, port_no, default_network_id):
    dp = self.setdefault_dpid(dpid)
    return dp.setdefault(port_no, Port(port_no=port_no, network_id=default_network_id))