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
def Ancestor(self, ancestor):
    """Sets an ancestor for this query.

    This restricts the query to only return result entities that are descended
    from a given entity. In other words, all of the results will have the
    ancestor as their parent, or parent's parent, or etc.

    Raises BadArgumentError or BadKeyError if parent is not an existing Entity
    or Key in the datastore.

    Args:
      # the key must be complete
      ancestor: Entity or Key

    Returns:
      # this query
      Query
    """
    self.__ancestor_pb = _GetCompleteKeyOrError(ancestor)._ToPb()
    return self