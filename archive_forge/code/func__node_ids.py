import json
import base64
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState, NodeDriver, NodeLocation
from libcloud.compute.types import Provider
from libcloud.common.upcloud import (
def _node_ids(self):
    """
        Returns list of server uids currently on upcloud
        """
    response = self.connection.request('1.2/server')
    servers = response.object['servers']['server']
    return [server['uuid'] for server in servers]