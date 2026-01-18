import collections
from os_ken.base import app_manager
import os_ken.exception as os_ken_exc
from os_ken.controller import event
from os_ken.exception import NetworkNotFound, NetworkAlreadyExist
from os_ken.exception import PortAlreadyExist, PortNotFound, PortUnknown
class MacAddresses(dict):
    """network_id -> mac_address -> set of (dpid, port_no)"""

    def add_port(self, network_id, dpid, port_no, mac_address):
        mac2port = self.setdefault(network_id, MacToPort())
        mac2port.add_port(dpid, port_no, mac_address)

    def remove_port(self, network_id, dpid, port_no, mac_address):
        mac2port = self.get(network_id)
        if mac2port is None:
            return
        mac2port.remove_port(dpid, port_no, mac_address)
        if not mac2port:
            del self[network_id]

    def get_ports(self, network_id, mac_address):
        mac2port = self.get(network_id)
        if not mac2port:
            return set()
        return mac2port.get_ports(mac_address)