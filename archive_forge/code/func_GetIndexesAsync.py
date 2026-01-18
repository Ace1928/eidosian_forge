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
def GetIndexesAsync(**kwargs):
    """Asynchronously retrieves the application indexes and their states.

  Identical to GetIndexes() except returns an asynchronous object. Call
  get_result() on the return value to block on the call and get the results.
  """
    extra_hook = kwargs.pop('extra_hook', None)
    config = _GetConfigFromKwargs(kwargs)

    def local_extra_hook(result):
        if extra_hook:
            return extra_hook(result)
        return result
    return _GetConnection().async_get_indexes(config, local_extra_hook)