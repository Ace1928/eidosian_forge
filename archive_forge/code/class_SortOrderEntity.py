import heapq
import itertools
import logging
import os
import re
import sys
import threading  # Knowing full well that this is a usually a placeholder.
import traceback
from xml.sax import saxutils
from googlecloudsdk.core.util import encoding
from googlecloudsdk.third_party.appengine.api import apiproxy_stub_map
from googlecloudsdk.third_party.appengine.api import capabilities
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_query
from googlecloudsdk.third_party.appengine.datastore import datastore_rpc
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
class SortOrderEntity(object):
    """Allow entity comparisons using provided orderings.

    The iterator passed to the constructor is eventually consumed via
    calls to GetNext(), which generate new SortOrderEntity s with the
    same orderings.
    """

    def __init__(self, entity_iterator, orderings):
        """Ctor.

      Args:
        entity_iterator: an iterator of entities which will be wrapped.
        orderings: an iterable of (identifier, order) pairs. order
          should be either Query.ASCENDING or Query.DESCENDING.
      """
        self.__entity_iterator = entity_iterator
        self.__entity = None
        self.__min_max_value_cache = {}
        try:
            self.__entity = entity_iterator.next()
        except StopIteration:
            pass
        else:
            self.__orderings = orderings

    def __str__(self):
        return str(self.__entity)

    def GetEntity(self):
        """Gets the wrapped entity."""
        return self.__entity

    def GetNext(self):
        """Wrap and return the next entity.

      The entity is retrieved from the iterator given at construction time.
      """
        return MultiQuery.SortOrderEntity(self.__entity_iterator, self.__orderings)

    def CmpProperties(self, that):
        """Compare two entities and return their relative order.

      Compares self to that based on the current sort orderings and the
      key orders between them. Returns negative, 0, or positive depending on
      whether self is less, equal to, or greater than that. This
      comparison returns as if all values were to be placed in ascending order
      (highest value last).  Only uses the sort orderings to compare (ignores
       keys).

      Args:
        that: SortOrderEntity

      Returns:
        Negative if self < that
        Zero if self == that
        Positive if self > that
      """
        if not self.__entity:
            return cmp(self.__entity, that.__entity)
        for identifier, order in self.__orderings:
            value1 = self.__GetValueForId(self, identifier, order)
            value2 = self.__GetValueForId(that, identifier, order)
            result = cmp(value1, value2)
            if order == Query.DESCENDING:
                result = -result
            if result:
                return result
        return 0

    def __GetValueForId(self, sort_order_entity, identifier, sort_order):
        value = _GetPropertyValue(sort_order_entity.__entity, identifier)
        if isinstance(value, list):
            entity_key = sort_order_entity.__entity.key()
            if (entity_key, identifier) in self.__min_max_value_cache:
                value = self.__min_max_value_cache[entity_key, identifier]
            elif sort_order == Query.DESCENDING:
                value = min(value)
            else:
                value = max(value)
            self.__min_max_value_cache[entity_key, identifier] = value
        return value

    def __cmp__(self, that):
        """Compare self to that w.r.t. values defined in the sort order.

      Compare an entity with another, using sort-order first, then the key
      order to break ties. This can be used in a heap to have faster min-value
      lookup.

      Args:
        that: other entity to compare to
      Returns:
        negative: if self is less than that in sort order
        zero: if self is equal to that in sort order
        positive: if self is greater than that in sort order
      """
        property_compare = self.CmpProperties(that)
        if property_compare:
            return property_compare
        else:
            return cmp(self.__entity.key(), that.__entity.key())