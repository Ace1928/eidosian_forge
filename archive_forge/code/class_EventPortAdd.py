import logging
import warnings
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.controller import ofp_event
from os_ken.controller.handler import set_ev_cls
import os_ken.exception as os_ken_exc
from os_ken.lib.dpid import dpid_to_str
class EventPortAdd(EventPortBase):
    """
    An event class for switch port status "ADD" notification.

    This event is generated when a new port is added to a switch.
    For OpenFlow switches, one can get the same notification by observing
    os_ken.controller.ofp_event.EventOFPPortStatus.
    An instance has at least the following attributes.

    ========= =================================================================
    Attribute Description
    ========= =================================================================
    dp        A os_ken.controller.controller.Datapath instance of the switch
    port      port number
    ========= =================================================================
    """

    def __init__(self, dp, port):
        super(EventPortAdd, self).__init__(dp, port)