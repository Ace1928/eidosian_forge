import logging
import warnings
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.controller import ofp_event
from os_ken.controller.handler import set_ev_cls
import os_ken.exception as os_ken_exc
from os_ken.lib.dpid import dpid_to_str
class EventPortBase(EventDPBase):

    def __init__(self, dp, port):
        super(EventPortBase, self).__init__(dp)
        self.port = port