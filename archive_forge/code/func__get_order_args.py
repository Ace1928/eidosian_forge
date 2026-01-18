from oslo_serialization import jsonutils
from oslo_utils import timeutils
import uuid
from barbicanclient import base
from barbicanclient.tests import test_client
from barbicanclient.v1 import orders
def _get_order_args(self, order_data):
    order_args = jsonutils.loads(order_data)
    order_args.update(order_args.pop('meta'))
    order_args.pop('type')
    return order_args