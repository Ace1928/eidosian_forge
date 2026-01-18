from __future__ import absolute_import
from __future__ import unicode_literals
import base64
import collections
import pickle
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine._internal import six_subset
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_index
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.datastore import datastore_rpc
class CompositeOrder(Order):
    """An immutable class that represents a sequence of Orders.

  This class proactively flattens sub-orders that are of type CompositeOrder.
  For example:
    CompositeOrder([O1, CompositeOrder([02, 03]), O4])
  is equivalent to:
    CompositeOrder([O1, 02, 03, O4])
  """

    def __init__(self, orders):
        """Constructor.

    Args:
      orders: A list of Orders which are applied in order.
    """
        if not isinstance(orders, (list, tuple)):
            raise datastore_errors.BadArgumentError('orders argument should be list or tuple (%r)' % (orders,))
        super(CompositeOrder, self).__init__()
        flattened = []
        for order in orders:
            if isinstance(order, CompositeOrder):
                flattened.extend(order._orders)
            elif isinstance(order, Order):
                flattened.append(order)
            else:
                raise datastore_errors.BadArgumentError('orders argument should only contain Order (%r)' % (order,))
        self._orders = tuple(flattened)

    @property
    def orders(self):
        return self._orders

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, list(self.orders))

    @datastore_rpc._positional(1)
    def reversed(self, group_by=None):
        return CompositeOrder([order.reversed(group_by=group_by) for order in self._orders])

    def _get_prop_names(self):
        names = set()
        for order in self._orders:
            names |= order._get_prop_names()
        return names

    def _key(self, lhs_value_map):
        result = []
        for order in self._orders:
            result.append(order._key(lhs_value_map))
        return tuple(result)

    def _cmp(self, lhs_value_map, rhs_value_map):
        for order in self._orders:
            result = order._cmp(lhs_value_map, rhs_value_map)
            if result != 0:
                return result
        return 0

    def size(self):
        """Returns the number of sub-orders the instance contains."""
        return len(self._orders)

    def _to_pbs(self):
        """Returns an ordered list of internal only pb representations."""
        return [order._to_pb() for order in self._orders]

    def _to_pb_v1(self, adapter):
        """Returns an ordered list of googledatastore.PropertyOrder.

    Args:
      adapter: A datastore_rpc.AbstractAdapter
    """
        return [order._to_pb_v1(adapter) for order in self._orders]

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return super(CompositeOrder, self).__eq__(other)
        if len(self._orders) == 1:
            result = self._orders[0].__eq__(other)
            if result is NotImplemented and hasattr(other, '__eq__'):
                return other.__eq__(self._orders[0])
            return result
        return NotImplemented