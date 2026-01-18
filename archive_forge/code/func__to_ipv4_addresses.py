import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _to_ipv4_addresses(self, object):
    ipv4_address_elements = object.findall(fixxpath('ipv4', TYPES_URN))
    return [self._to_ipv4_6_address(el) for el in ipv4_address_elements]