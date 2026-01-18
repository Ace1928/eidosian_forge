import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _to_ip_address_list(self, element):
    ipAddresses = []
    for ip in findall(element, 'ipAddress', TYPES_URN):
        ipAddresses.append(self._to_ip_address(ip))
    child_ip_address_lists = []
    for child_ip_list in findall(element, 'childIpAddressList', TYPES_URN):
        child_ip_address_lists.append(self._to_child_ip_list(child_ip_list))
    return NttCisIpAddressList(id=element.get('id'), name=findtext(element, 'name', TYPES_URN), description=findtext(element, 'description', TYPES_URN), ip_version=findtext(element, 'ipVersion', TYPES_URN), ip_address_collection=ipAddresses, state=findtext(element, 'state', TYPES_URN), create_time=findtext(element, 'createTime', TYPES_URN), child_ip_address_lists=child_ip_address_lists)