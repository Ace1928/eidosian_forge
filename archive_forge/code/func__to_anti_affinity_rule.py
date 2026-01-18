import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _to_anti_affinity_rule(self, element):
    node_list = []
    for node in findall(element, 'serverSummary', TYPES_URN):
        node_list.append(node.get('id'))
    return NttCisAntiAffinityRule(id=element.get('id'), node_list=node_list)