import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _to_ip_address_lists(self, object):
    ip_address_lists = []
    for element in findall(object, 'ipAddressList', TYPES_URN):
        ip_address_lists.append(self._to_ip_address_list(element))
    return ip_address_lists