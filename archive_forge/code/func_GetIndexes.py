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
def GetIndexes(**kwargs):
    """Retrieves the application indexes and their states.

  Args:
    config: Optional Configuration to use for this request, must be specified
      as a keyword argument.

  Returns:
    A list of (Index, Index.[BUILDING|SERVING|DELETING|ERROR]) tuples.
    An index can be in the following states:
      Index.BUILDING: Index is being built and therefore can not serve queries
      Index.SERVING: Index is ready to service queries
      Index.DELETING: Index is being deleted
      Index.ERROR: Index encounted an error in the BUILDING state
  """
    return GetIndexesAsync(**kwargs).get_result()