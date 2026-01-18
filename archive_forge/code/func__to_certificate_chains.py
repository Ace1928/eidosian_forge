from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def _to_certificate_chains(self, object):
    cert_chains = []
    for element in object.findall(fixxpath('sslCertificateChain', TYPES_URN)):
        cert_chains.append(self._to_certificate_chain(element))
    return cert_chains