import collections
from os_ken.base import app_manager
import os_ken.exception as os_ken_exc
from os_ken.controller import event
from os_ken.exception import NetworkNotFound, NetworkAlreadyExist
from os_ken.exception import PortAlreadyExist, PortNotFound, PortUnknown
def _check_nw_id_unknown(self, network_id):
    if network_id == self.nw_id_unknown:
        raise NetworkAlreadyExist(network_id=network_id)