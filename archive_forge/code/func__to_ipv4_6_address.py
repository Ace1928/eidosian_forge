import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _to_ipv4_6_address(self, element):
    return NttCisReservedIpAddress(element.get('datacenterId'), element.get('exclusive'), findtext(element, 'vlanId', TYPES_URN), findtext(element, 'ipAddress', TYPES_URN), description=findtext(element, 'description', TYPES_URN))