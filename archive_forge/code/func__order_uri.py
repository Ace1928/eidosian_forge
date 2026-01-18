from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import Provider, NodeState
def _order_uri(self, node, resource):
    return '/orders/{}/{}'.format(node.id, resource)