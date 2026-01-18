import json
import base64
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState, NodeDriver, NodeLocation
from libcloud.compute.types import Provider
from libcloud.common.upcloud import (
def _to_node_sizes(self, plans, prices, location):
    plan_price = PlanPrice(prices)
    return [self._to_node_size(plan, plan_price, location) for plan in plans]