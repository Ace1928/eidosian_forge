import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _to_ip_block(self, element, locations):
    location_id = element.get('datacenterId')
    location = list(filter(lambda x: x.id == location_id, locations))[0]
    return NttCisPublicIpBlock(id=element.get('id'), network_domain=self.ex_get_network_domain(findtext(element, 'networkDomainId', TYPES_URN)), base_ip=findtext(element, 'baseIp', TYPES_URN), size=findtext(element, 'size', TYPES_URN), location=location, status=findtext(element, 'state', TYPES_URN))