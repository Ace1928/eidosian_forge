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
def GetFilterPredicate(self):
    """Returns a datastore_query.FilterPredicate for the current instance.

    Returns:
      datastore_query.FilterPredicate or None if no filters are set on the
      current Query.
    """
    ordered_filters = [(i, f) for f, i in self.__filter_order.iteritems()]
    ordered_filters.sort()
    property_filters = []
    for _, filter_str in ordered_filters:
        if filter_str not in self:
            continue
        values = self[filter_str]
        match = self._CheckFilter(filter_str, values)
        name = match.group(1)
        op = match.group(3)
        if op is None or op == '==':
            op = '='
        property_filters.append(datastore_query.make_filter(name, op, values))
    if property_filters:
        return datastore_query.CompositeFilter(datastore_query.CompositeFilter.AND, property_filters)
    return None