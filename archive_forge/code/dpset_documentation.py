import logging
import warnings
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.controller import ofp_event
from os_ken.controller.handler import set_ev_cls
import os_ken.exception as os_ken_exc
from os_ken.lib.dpid import dpid_to_str

        This method returns a list of os_ken.controller.dpset.PortState
        instances for the given Datapath ID.
        Raises KeyError if no such a datapath connected to this controller.
        