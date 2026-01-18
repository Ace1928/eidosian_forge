from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def _to_certs(self, object):
    certs = []
    for element in object.findall(fixxpath('sslDomainCertificate', TYPES_URN)):
        certs.append(self._to_cert(element))
    return certs