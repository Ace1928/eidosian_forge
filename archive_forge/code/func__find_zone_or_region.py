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
def _find_zone_or_region(self, name, res_type, region=False, res_name=None):
    """
        Find the zone or region for a named resource.

        :param  name: Name of resource to find
        :type   name: ``str``

        :param  res_type: Type of resource to find.
                          Examples include: 'disks', 'instances' or 'addresses'
        :type   res_type: ``str``

        :keyword  region: If True, search regions instead of zones
        :type     region: ``bool``

        :keyword  res_name: The name of the resource type for error messages.
                            Examples: 'Volume', 'Node', 'Address'
        :keyword  res_name: ``str``

        :return:  Zone/Region object for the zone/region for the resource.
        :rtype:   :class:`GCEZone` or :class:`GCERegion`
        """
    if region:
        rz = 'region'
    else:
        rz = 'zone'
    rz_name = None
    res_name = res_name or res_type
    res_list = self.connection.request_aggregated_items(res_type)
    for k, v in res_list['items'].items():
        for res in v.get(res_type, []):
            if res['name'] == name:
                rz_name = k.replace('%ss/' % rz, '')
                break
    if not rz_name:
        raise ResourceNotFoundError("{} '{}' not found in any {}.".format(res_name, name, rz), None, None)
    else:
        getrz = getattr(self, 'ex_get_%s' % rz)
        return getrz(rz_name)