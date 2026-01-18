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