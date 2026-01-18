import collections
import logging
import os_ken.exception as os_ken_exc
from os_ken.base import app_manager
from os_ken.controller import event
class TunnelKeys(dict):
    """network id(uuid) <-> tunnel key(32bit unsigned int)"""

    def __init__(self, f):
        super(TunnelKeys, self).__init__()
        self.send_event = f

    def get_key(self, network_id):
        try:
            return self[network_id]
        except KeyError:
            raise TunnelKeyNotFound(network_id=network_id)

    def _set_key(self, network_id, tunnel_key):
        self[network_id] = tunnel_key
        self.send_event(EventTunnelKeyAdd(network_id, tunnel_key))

    def register_key(self, network_id, tunnel_key):
        if network_id in self:
            raise os_ken_exc.NetworkAlreadyExist(network_id=network_id)
        if tunnel_key in self.values():
            raise TunnelKeyAlreadyExist(tunnel_key=tunnel_key)
        self._set_key(network_id, tunnel_key)

    def update_key(self, network_id, tunnel_key):
        if network_id not in self and tunnel_key in self.values():
            raise TunnelKeyAlreadyExist(key=tunnel_key)
        key = self.get(network_id)
        if key is None:
            self._set_key(network_id, tunnel_key)
            return
        if key != tunnel_key:
            raise os_ken_exc.NetworkAlreadyExist(network_id=network_id)

    def delete_key(self, network_id):
        try:
            tunnel_key = self[network_id]
            self.send_event(EventTunnelKeyDel(network_id, tunnel_key))
            del self[network_id]
        except KeyError:
            raise os_ken_exc.NetworkNotFound(network_id=network_id)