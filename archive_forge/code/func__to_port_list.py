import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _to_port_list(self, element):
    ports = []
    for port in findall(element, 'port', TYPES_URN):
        ports.append(self._to_port(element=port))
    child_portlist_list = []
    for child in findall(element, 'childPortList', TYPES_URN):
        child_portlist_list.append(self._to_child_port_list(element=child))
    return NttCisPortList(id=element.get('id'), name=findtext(element, 'name', TYPES_URN), description=findtext(element, 'description', TYPES_URN), port_collection=ports, child_portlist_list=child_portlist_list, state=findtext(element, 'state', TYPES_URN), create_time=findtext(element, 'createTime', TYPES_URN))