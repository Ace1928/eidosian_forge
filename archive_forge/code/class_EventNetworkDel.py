import collections
from os_ken.base import app_manager
import os_ken.exception as os_ken_exc
from os_ken.controller import event
from os_ken.exception import NetworkNotFound, NetworkAlreadyExist
from os_ken.exception import PortAlreadyExist, PortNotFound, PortUnknown
class EventNetworkDel(event.EventBase):
    """
    An event class for network deletion.

    This event is generated when a network is deleted by the REST API.
    An instance has at least the following attributes.

    ========== ===================================================================
    Attribute  Description
    ========== ===================================================================
    network_id Network ID
    ========== ===================================================================
    """

    def __init__(self, network_id):
        super(EventNetworkDel, self).__init__()
        self.network_id = network_id