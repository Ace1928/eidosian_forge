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
def _to_healthcheck(self, healthcheck):
    """
        Return a HealthCheck object from the JSON-response dictionary.

        :param  healthcheck: The dictionary describing the healthcheck.
        :type   healthcheck: ``dict``

        :return: HealthCheck object
        :rtype: :class:`GCEHealthCheck`
        """
    extra = {}
    extra['selfLink'] = healthcheck.get('selfLink')
    extra['creationTimestamp'] = healthcheck.get('creationTimestamp')
    extra['description'] = healthcheck.get('description')
    extra['host'] = healthcheck.get('host')
    return GCEHealthCheck(id=healthcheck['id'], name=healthcheck['name'], path=healthcheck.get('requestPath'), port=healthcheck.get('port'), interval=healthcheck.get('checkIntervalSec'), timeout=healthcheck.get('timeoutSec'), unhealthy_threshold=healthcheck.get('unhealthyThreshold'), healthy_threshold=healthcheck.get('healthyThreshold'), driver=self, extra=extra)