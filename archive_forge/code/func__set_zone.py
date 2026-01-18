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
def _set_zone(self, zone):
    """
        Return the zone to use for listing resources.

        :param  zone: A name, zone object, None, or 'all'
        :type   zone: ``str`` or :class:`GCEZone` or ``None``

        :return:  A zone object or None if all zones should be considered
        :rtype:   :class:`GCEZone` or ``None``
        """
    zone = zone or self.zone
    if zone == 'all' or zone is None:
        return None
    if not hasattr(zone, 'name'):
        zone = self.ex_get_zone(zone)
    return zone