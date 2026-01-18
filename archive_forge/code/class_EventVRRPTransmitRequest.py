from os_ken.controller import handler
from os_ken.controller import event
from os_ken.lib import dpid as dpid_lib
from os_ken.lib import mac as mac_lib
from os_ken.lib.packet import vrrp
from os_ken.lib import addrconv
class EventVRRPTransmitRequest(event.EventRequestBase):
    """
    Request from VRRP router to port manager to transmit VRRP packet.
    """

    def __init__(self, data):
        super(EventVRRPTransmitRequest, self).__init__()
        self.data = data