import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _to_cpu_spec(self, element):
    return NttCisServerCpuSpecification(cpu_count=int(element.get('count')), cores_per_socket=int(element.get('coresPerSocket')), performance=element.get('speed'))