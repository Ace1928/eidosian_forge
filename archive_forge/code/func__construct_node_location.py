import json
import base64
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState, NodeDriver, NodeLocation
from libcloud.compute.types import Provider
from libcloud.common.upcloud import (
def _construct_node_location(self, zone):
    return NodeLocation(id=zone['id'], name=zone['description'], country=self._parse_country(zone['id']), driver=self)