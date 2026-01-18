import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _to_port_lists(self, object):
    port_lists = []
    for element in findall(object, 'portList', TYPES_URN):
        port_lists.append(self._to_port_list(element))
    return port_lists