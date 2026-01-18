import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _to_network_domains(self, object):
    network_domains = []
    locations = self.list_locations()
    for element in findall(object, 'networkDomain', TYPES_URN):
        network_domains.append(self._to_network_domain(element, locations))
    return network_domains