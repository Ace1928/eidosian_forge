import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _to_firewall_rules(self, object, network_domain):
    rules = []
    locations = self.list_locations()
    for element in findall(object, 'firewallRule', TYPES_URN):
        rules.append(self._to_firewall_rule(element, locations, network_domain))
    return rules