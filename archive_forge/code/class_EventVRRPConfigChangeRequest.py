from os_ken.controller import handler
from os_ken.controller import event
from os_ken.lib import dpid as dpid_lib
from os_ken.lib import mac as mac_lib
from os_ken.lib.packet import vrrp
from os_ken.lib import addrconv
class EventVRRPConfigChangeRequest(event.EventRequestBase):
    """
    Event that requests to change configuration of a given VRRP router.
    None means no-change.
    """

    def __init__(self, instance_name, priority=None, advertisement_interval=None, preempt_mode=None, preempt_delay=None, accept_mode=None):
        super(EventVRRPConfigChangeRequest, self).__init__()
        self.instance_name = instance_name
        self.priority = priority
        self.advertisement_interval = advertisement_interval
        self.preempt_mode = preempt_mode
        self.preempt_delay = preempt_delay
        self.accept_mode = accept_mode