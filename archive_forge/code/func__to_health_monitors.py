from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def _to_health_monitors(self, object):
    monitors = []
    matches = object.findall(fixxpath('defaultHealthMonitor', TYPES_URN))
    for element in matches:
        monitors.append(self._to_health_monitor(element))
    return monitors