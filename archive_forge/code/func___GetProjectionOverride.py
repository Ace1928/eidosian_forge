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
def __GetProjectionOverride(self, config):
    """Returns a tuple of (original projection, projection override).

    If projection is None, there is no projection. If override is None,
    projection is sufficent for this query.
    """
    projection = datastore_query.QueryOptions.projection(config)
    if projection is None:
        projection = self.__projection
    else:
        projection = projection
    if not projection:
        return (None, None)
    override = set()
    for prop, _ in self.__orderings:
        if prop not in projection:
            override.add(prop)
    if not override:
        return (projection, None)
    return (projection, projection + tuple(override))