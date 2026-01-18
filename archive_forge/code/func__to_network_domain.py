import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _to_network_domain(self, element, locations):
    location_id = element.get('datacenterId')
    location = list(filter(lambda x: x.id == location_id, locations))[0]
    plan = findtext(element, 'type', TYPES_URN)
    if plan == 'ESSENTIALS':
        plan_type = NetworkDomainServicePlan.ESSENTIALS
    else:
        plan_type = NetworkDomainServicePlan.ADVANCED
    return NttCisNetworkDomain(id=element.get('id'), name=findtext(element, 'name', TYPES_URN), description=findtext(element, 'description', TYPES_URN), plan=plan_type, location=location, status=findtext(element, 'state', TYPES_URN))