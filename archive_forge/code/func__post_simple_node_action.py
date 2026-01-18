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
def _post_simple_node_action(self, node, action):
    """Post a simple, data-less action to the OS node action endpoint
        :param `Node` node:
        :param str action: the action to call
        :return `bool`: a boolean that indicates success
        """
    uri = '/servers/{node_id}/action'.format(node_id=node.id)
    resp = self.connection.request(uri, method='POST', data={action: None})
    return resp.status == httplib.ACCEPTED