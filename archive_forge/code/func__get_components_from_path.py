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
def _get_components_from_path(self, path):
    """
        Return a dictionary containing name & zone/region from a request path.

        :param  path: HTTP request path (e.g.
                      '/project/pjt-name/zones/us-central1-a/instances/mynode')
        :type   path: ``str``

        :return:  Dictionary containing name and zone/region of resource
        :rtype:   ``dict``
        """
    region = None
    zone = None
    glob = False
    components = path.split('/')
    name = components[-1]
    if components[-4] == 'regions':
        region = components[-3]
    elif components[-4] == 'zones':
        zone = components[-3]
    elif components[-3] == 'global':
        glob = True
    return {'name': name, 'region': region, 'zone': zone, 'global': glob}