import inspect
import time
from os_ken.controller import handler
from os_ken import ofproto
from . import event
class EventOFPStateChange(event.EventBase):
    """
    An event class for negotiation phase change notification.

    An instance of this class is sent to observer after changing
    the negotiation phase.
    An instance has at least the following attributes.

    ========= =================================================================
    Attribute Description
    ========= =================================================================
    datapath  os_ken.controller.controller.Datapath instance of the switch
    ========= =================================================================
    """

    def __init__(self, dp):
        super(EventOFPStateChange, self).__init__()
        self.datapath = dp