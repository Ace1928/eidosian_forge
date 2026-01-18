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
def _to_targethttpproxy(self, targethttpproxy):
    """
        Return a Target HTTP Proxy object from the JSON-response dictionary.

        :param  targethttpproxy: The dictionary describing the proxy.
        :type   targethttpproxy: ``dict``

        :return: Target HTTP Proxy object
        :rtype:  :class:`GCETargetHttpProxy`
        """
    extra = {k: targethttpproxy.get(k) for k in ('creationTimestamp', 'description', 'selfLink')}
    urlmap = self._get_object_by_kind(targethttpproxy.get('urlMap'))
    return GCETargetHttpProxy(id=targethttpproxy['id'], name=targethttpproxy['name'], urlmap=urlmap, driver=self, extra=extra)