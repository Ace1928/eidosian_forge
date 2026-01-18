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
def _to_shared_ip_group(self, el):
    servers_el = findall(el, 'servers', self.XML_NAMESPACE)
    if servers_el:
        servers = [s.get('id') for s in findall(servers_el[0], 'server', self.XML_NAMESPACE)]
    else:
        servers = None
    return OpenStack_1_0_SharedIpGroup(id=el.get('id'), name=el.get('name'), servers=servers)