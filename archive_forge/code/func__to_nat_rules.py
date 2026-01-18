import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _to_nat_rules(self, object, network_domain):
    rules = []
    for element in findall(object, 'natRule', TYPES_URN):
        rules.append(self._to_nat_rule(element, network_domain))
    return rules