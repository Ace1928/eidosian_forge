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
def _manage_router_interface(self, router, op, subnet=None, port=None):
    """
        Add/Remove interface to router

        :param router: Router to add/remove the interface
        :type router: :class:`OpenStack_2_Router`

        :param      op: Operation to perform: 'add' or 'remove'
        :type       op: ``str``

        :param subnet: Subnet object to be added to the router
        :type subnet: :class:`OpenStack_2_SubNet`

        :param port: Port object to be added to the router
        :type port: :class:`OpenStack_2_PortInterface`

        :rtype: ``bool``
        """
    data = {}
    if subnet:
        data['subnet_id'] = subnet.id
    elif port:
        data['port_id'] = port.id
    else:
        raise OpenStackException('Error in router interface: port or subnet are None.', 500, self)
    resp = self.network_connection.request('{}/{}/{}_router_interface'.format('/v2.0/routers', router.id, op), method='PUT', data=data)
    return resp.status == httplib.OK