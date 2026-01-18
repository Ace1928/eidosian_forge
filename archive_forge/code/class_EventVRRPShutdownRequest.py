from os_ken.controller import handler
from os_ken.controller import event
from os_ken.lib import dpid as dpid_lib
from os_ken.lib import mac as mac_lib
from os_ken.lib.packet import vrrp
from os_ken.lib import addrconv
class EventVRRPShutdownRequest(event.EventRequestBase):
    """
    Request from management layer to VRRP to shutdown VRRP Router.
    """

    def __init__(self, instance_name):
        super(EventVRRPShutdownRequest, self).__init__()
        self.instance_name = instance_name