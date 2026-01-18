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
def _to_region(self, region):
    """
        Return a Region object from the JSON-response dictionary.

        :param  region: The dictionary describing the region.
        :type   region: ``dict``

        :return: Region object
        :rtype: :class:`GCERegion`
        """
    extra = {}
    extra['selfLink'] = region.get('selfLink')
    extra['creationTimestamp'] = region.get('creationTimestamp')
    extra['description'] = region.get('description')
    quotas = region.get('quotas')
    zones = [self.ex_get_zone(z) for z in region.get('zones', [])]
    zones = [z for z in zones if z is not None]
    deprecated = region.get('deprecated')
    return GCERegion(id=region['id'], name=region['name'], status=region.get('status'), zones=zones, quotas=quotas, deprecated=deprecated, driver=self, extra=extra)