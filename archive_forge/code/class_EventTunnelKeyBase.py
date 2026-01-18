import collections
import logging
import os_ken.exception as os_ken_exc
from os_ken.base import app_manager
from os_ken.controller import event
class EventTunnelKeyBase(event.EventBase):

    def __init__(self, network_id, tunnel_key):
        super(EventTunnelKeyBase, self).__init__()
        self.network_id = network_id
        self.tunnel_key = tunnel_key