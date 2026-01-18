import json
from libcloud.compute.base import (
from libcloud.common.gig_g8 import G8Connection
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
class G8Network(UuidMixin):
    """
    G8 Network object class.

    This class maps to a cloudspace

    """

    def __init__(self, id, name, cidr, publicipaddress, driver, extra=None):
        self.id = id
        self.name = name
        self._cidr = cidr
        self.driver = driver
        self.publicipaddress = publicipaddress
        self.extra = extra
        UuidMixin.__init__(self)

    @property
    def cidr(self):
        """
        Cidr is not part of the list result
        we will lazily fetch it with a get request
        """
        if self._cidr is None:
            networkdata = self.driver._api_request('/cloudspaces/get', {'cloudspaceId': self.id})
            self._cidr = networkdata['privatenetwork']
        return self._cidr

    def list_nodes(self):
        return self.driver.list_nodes(self)

    def destroy(self):
        return self.driver.ex_destroy_network(self)

    def list_portforwards(self):
        return self.driver.ex_list_portforwards(self)

    def create_portforward(self, node, publicport, privateport, protocol='tcp'):
        return self.driver.ex_create_portforward(self, node, publicport, privateport, protocol)