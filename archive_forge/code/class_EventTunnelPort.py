import collections
import logging
import os_ken.exception as os_ken_exc
from os_ken.base import app_manager
from os_ken.controller import event
class EventTunnelPort(event.EventBase):
    """
    An event class for tunnel port registration.

    This event is generated when a tunnel port is added or removed
    by the REST API.
    An instance has at least the following attributes.

    =========== ===============================================================
    Attribute   Description
    =========== ===============================================================
    dpid        OpenFlow Datapath ID
    port_no     OpenFlow port number
    remote_dpid OpenFlow port number of the tunnel peer
    add_del     True for adding a tunnel.  False for removal.
    =========== ===============================================================
    """

    def __init__(self, dpid, port_no, remote_dpid, add_del):
        super(EventTunnelPort, self).__init__()
        self.dpid = dpid
        self.port_no = port_no
        self.remote_dpid = remote_dpid
        self.add_del = add_del