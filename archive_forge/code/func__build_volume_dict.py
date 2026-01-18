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
def _build_volume_dict(self, zone_dict):
    """
        Build a dictionary in [name][zone]=disk format.

        :param  zone_dict: dict in the format of:
                 { items: {key: {api_name:[], key2: api_name:[]}} }
        :type   zone_dict: ``dict``

        :return:  dict of volumes, organized by name, then zone  Format:
                  { 'disk_name':
                   {'zone_name1': disk_info, 'zone_name2': disk_info} }
        :rtype: ``dict``
        """
    name_zone_dict = {}
    for k, v in zone_dict.items():
        zone_name = k.replace('zones/', '')
        disks = v.get('disks', [])
        for disk in disks:
            n = disk['name']
            name_zone_dict.setdefault(n, {})
            name_zone_dict[n].update({zone_name: disk})
    return name_zone_dict