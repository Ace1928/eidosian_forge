from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def _to_persistence_profile(self, element):
    compatible = []
    matches = element.findall(fixxpath('virtualListenerCompatibility', TYPES_URN))
    for match_element in matches:
        compatible.append(NttCisVirtualListenerCompatibility(type=match_element.get('type'), protocol=match_element.get('protocol', None)))
    return NttCisPersistenceProfile(id=element.get('id'), fallback_compatible=bool(element.get('fallbackCompatible') == 'true'), name=findtext(element, 'name', TYPES_URN), compatible_listeners=compatible)