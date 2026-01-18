import base64
import warnings
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, next, httplib, parse_qs, urlparse
from libcloud.utils.xml import findall
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.openstack import (
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
class OpenStack_1_1_FloatingIpAddress:
    """
    Floating IP info.
    """

    def __init__(self, id, ip_address, pool, node_id=None, driver=None):
        self.id = str(id)
        self.ip_address = ip_address
        self.pool = pool
        self.node_id = node_id
        self.driver = driver

    def delete(self):
        """
        Delete this floating IP

        :rtype: ``bool``
        """
        if self.pool is not None:
            return self.pool.delete_floating_ip(self)
        elif self.driver is not None:
            return self.driver.ex_delete_floating_ip(self)

    def __repr__(self):
        return '<OpenStack_1_1_FloatingIpAddress: id=%s, ip_addr=%s, pool=%s, driver=%s>' % (self.id, self.ip_address, self.pool, self.driver)