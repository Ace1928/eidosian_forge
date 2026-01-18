from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def _to_pools(self, object):
    pools = []
    for element in object.findall(fixxpath('pool', TYPES_URN)):
        pools.append(self._to_pool(element))
    return pools