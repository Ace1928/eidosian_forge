from os_ken.controller import handler
from os_ken.controller import event
from os_ken.lib import dpid as dpid_lib
from os_ken.lib import mac as mac_lib
from os_ken.lib.packet import vrrp
from os_ken.lib import addrconv
class EventVRRPConfigReply(event.EventReplyBase):

    def __init__(self, instance_name, interface, config):
        super(EventVRRPConfigReply, self).__init__(None)
        self.instance_name = instance_name
        self.interface = interface
        self.config = config