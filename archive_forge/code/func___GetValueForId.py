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