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
def _multi_create_disk(self, status, node_attrs):
    """Create disk for ex_create_multiple_nodes.

        :param  status: Dictionary for holding node/disk creation status.
                        (This dictionary is modified by this method)
        :type   status: ``dict``

        :param  node_attrs: Dictionary for holding node attribute information.
                            (size, image, location, ex_disk_type, etc.)
        :type   node_attrs: ``dict``
        """
    disk = None
    if node_attrs['use_existing_disk']:
        try:
            disk = self.ex_get_volume(status['name'], node_attrs['location'])
        except ResourceNotFoundError:
            pass
    if disk:
        status['disk'] = disk
    else:
        disk_req, disk_data, disk_params = self._create_vol_req(None, status['name'], location=node_attrs['location'], image=node_attrs['image'], ex_disk_type=node_attrs['ex_disk_type'])
        try:
            disk_res = self.connection.request(disk_req, method='POST', data=disk_data, params=disk_params).object
        except GoogleBaseError:
            e = self._catch_error(ignore_errors=node_attrs['ignore_errors'])
            error = e.value
            code = e.code
            disk_res = None
            status['disk'] = GCEFailedDisk(status['name'], error, code)
        status['disk_response'] = disk_res