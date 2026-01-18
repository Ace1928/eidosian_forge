import json
import base64
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState, NodeDriver, NodeLocation
from libcloud.compute.types import Provider
from libcloud.common.upcloud import (
def _to_node_locations(self, zones):
    return [self._construct_node_location(zone) for zone in zones]