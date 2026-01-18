import base64
from libcloud.utils.py3 import b
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError, MalformedResponseError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import NodeState
from libcloud.compute.providers import Provider
def _get_price_per_node(self, num):
    single_node_price = self._get_size_price(size_id='1')
    return num * single_node_price