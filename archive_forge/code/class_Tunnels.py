import collections
import logging
import os_ken.exception as os_ken_exc
from os_ken.base import app_manager
from os_ken.controller import event
class Tunnels(app_manager.OSKenApp):

    def __init__(self):
        super(Tunnels, self).__init__()
        self.name = 'tunnels'
        self.tunnel_keys = TunnelKeys(self.send_event_to_observers)
        self.dpids = DPIDs(self.send_event_to_observers)

    def get_key(self, network_id):
        return self.tunnel_keys.get_key(network_id)

    def register_key(self, network_id, tunnel_key):
        self.tunnel_keys.register_key(network_id, tunnel_key)

    def update_key(self, network_id, tunnel_key):
        self.tunnel_keys.update_key(network_id, tunnel_key)

    def delete_key(self, network_id):
        self.tunnel_keys.delete_key(network_id)

    def list_ports(self, dpid):
        return self.dpids.list_ports(dpid).keys()

    def register_port(self, dpid, port_no, remote_dpid):
        self.dpids.add_remote_dpid(dpid, port_no, remote_dpid)

    def update_port(self, dpid, port_no, remote_dpid):
        self.dpids.update_remote_dpid(dpid, port_no, remote_dpid)

    def get_remote_dpid(self, dpid, port_no):
        return self.dpids.get_remote_dpid(dpid, port_no)

    def delete_port(self, dpid, port_no):
        self.dpids.delete_port(dpid, port_no)

    def get_port(self, dpid, remote_dpid):
        return self.dpids.get_port(dpid, remote_dpid)