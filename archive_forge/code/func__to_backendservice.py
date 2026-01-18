import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def _to_backendservice(self, backendservice):
    """
        Return a Backend Service object from the JSON-response dictionary.

        :param  backendservice: The dictionary describing the backend service.
        :type   backendservice: ``dict``

        :return: BackendService object
        :rtype: :class:`GCEBackendService`
        """
    extra = {}
    for extra_key in ('selfLink', 'creationTimestamp', 'fingerprint', 'description'):
        extra[extra_key] = backendservice.get(extra_key)
    backends = backendservice.get('backends', [])
    healthchecks = [self._get_object_by_kind(h) for h in backendservice.get('healthChecks', [])]
    return GCEBackendService(id=backendservice['id'], name=backendservice['name'], backends=backends, healthchecks=healthchecks, port=backendservice['port'], port_name=backendservice['portName'], protocol=backendservice['protocol'], timeout=backendservice['timeoutSec'], driver=self, extra=extra)