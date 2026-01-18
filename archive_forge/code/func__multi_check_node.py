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
def _multi_check_node(self, status, node_attrs):
    """Check node status for ex_create_multiple_nodes.

        :param  status: Dictionary for holding node/disk creation status.
                        (This dictionary is modified by this method)
        :type   status: ``dict``

        :param  node_attrs: Dictionary for holding node attribute information.
                            (size, image, location, etc.)
        :type   node_attrs: ``dict``
        """
    error = None
    code = None
    try:
        response = self.connection.request(status['node_response']['selfLink']).object
    except GoogleBaseError:
        e = self._catch_error(ignore_errors=node_attrs['ignore_errors'])
        error = e.value
        code = e.code
        response = {'status': 'DONE'}
    if response['status'] == 'DONE':
        status['node_response'] = None
        if error:
            status['node'] = GCEFailedNode(status['name'], error, code)
        else:
            status['node'] = self.ex_get_node(status['name'], node_attrs['location'])