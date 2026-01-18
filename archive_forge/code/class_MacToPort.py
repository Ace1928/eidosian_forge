import collections
from os_ken.base import app_manager
import os_ken.exception as os_ken_exc
from os_ken.controller import event
from os_ken.exception import NetworkNotFound, NetworkAlreadyExist
from os_ken.exception import PortAlreadyExist, PortNotFound, PortUnknown
class MacToPort(collections.defaultdict):
    """mac_address -> set of MacPort(dpid, port_no)"""

    def __init__(self):
        super(MacToPort, self).__init__(set)

    def add_port(self, dpid, port_no, mac_address):
        self[mac_address].add(MacPort(dpid, port_no))

    def remove_port(self, dpid, port_no, mac_address):
        ports = self[mac_address]
        ports.discard(MacPort(dpid, port_no))
        if not ports:
            del self[mac_address]

    def get_ports(self, mac_address):
        return self[mac_address]