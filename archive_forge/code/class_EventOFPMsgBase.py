import inspect
import time
from os_ken.controller import handler
from os_ken import ofproto
from . import event
class EventOFPMsgBase(event.EventBase):
    """
    The base class of OpenFlow event class.

    OpenFlow event classes have at least the following attributes.

    .. tabularcolumns:: |l|L|

    ============ ==============================================================
    Attribute    Description
    ============ ==============================================================
    msg          An object which describes the corresponding OpenFlow message.
    msg.datapath A os_ken.controller.controller.Datapath instance
                 which describes an OpenFlow switch from which we received
                 this OpenFlow message.
    timestamp    Timestamp when Datapath instance generated this event.
    ============ ==============================================================

    The msg object has some more additional members whose values are extracted
    from the original OpenFlow message.
    """

    def __init__(self, msg):
        self.timestamp = time.time()
        super(EventOFPMsgBase, self).__init__()
        self.msg = msg