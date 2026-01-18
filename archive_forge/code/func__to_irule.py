from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def _to_irule(self, element):
    compatible = []
    matches = element.findall(fixxpath('virtualListenerCompatibility', TYPES_URN))
    for match_element in matches:
        compatible.append(NttCisVirtualListenerCompatibility(type=match_element.get('type'), protocol=match_element.get('protocol', None)))
    irule_element = element.find(fixxpath('irule', TYPES_URN))
    return NttCisDefaultiRule(id=irule_element.get('id'), name=irule_element.get('name'), compatible_listeners=compatible)