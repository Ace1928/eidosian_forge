from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def _to_persistence_profiles(self, object):
    profiles = []
    matches = object.findall(fixxpath('defaultPersistenceProfile', TYPES_URN))
    for element in matches:
        profiles.append(self._to_persistence_profile(element))
    return profiles