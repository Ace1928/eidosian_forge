from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def _to_ssl_profiles(self, object):
    profiles = []
    for element in object.findall(fixxpath('sslOffloadProfile', TYPES_URN)):
        profiles.append(self._to_ssl_profile(element))
    return profiles