import logging
import warnings
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.controller import ofp_event
from os_ken.controller.handler import set_ev_cls
import os_ken.exception as os_ken_exc
from os_ken.lib.dpid import dpid_to_str
class EventDP(EventDPBase):
    """
    An event class to notify connect/disconnect of a switch.

    For OpenFlow switches, one can get the same notification by observing
    os_ken.controller.ofp_event.EventOFPStateChange.
    An instance has at least the following attributes.

    ========= =================================================================
    Attribute Description
    ========= =================================================================
    dp        A os_ken.controller.controller.Datapath instance of the switch
    enter     True when the switch connected to our controller.  False for
              disconnect.
    ports     A list of port instances.
    ========= =================================================================
    """

    def __init__(self, dp, enter_leave):
        super(EventDP, self).__init__(dp)
        self.enter = enter_leave
        self.ports = []