import collections
from os_ken.base import app_manager
import os_ken.exception as os_ken_exc
from os_ken.controller import event
from os_ken.exception import NetworkNotFound, NetworkAlreadyExist
from os_ken.exception import PortAlreadyExist, PortNotFound, PortUnknown
class EventNetworkPort(event.EventBase):
    """
    An event class for notification of port arrival and deperture.

    This event is generated when a port is introduced to or removed from a
    network by the REST API.
    An instance has at least the following attributes.

    ========== ================================================================
    Attribute  Description
    ========== ================================================================
    network_id Network ID
    dpid       OpenFlow Datapath ID of the switch to which the port belongs.
    port_no    OpenFlow port number of the port
    add_del    True for adding a port.  False for removing a port.
    ========== ================================================================
    """

    def __init__(self, network_id, dpid, port_no, add_del):
        super(EventNetworkPort, self).__init__()
        self.network_id = network_id
        self.dpid = dpid
        self.port_no = port_no
        self.add_del = add_del