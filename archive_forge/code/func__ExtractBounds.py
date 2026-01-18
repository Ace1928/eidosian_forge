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
def _ExtractBounds(self, config):
    """This function extracts the range of results to consider.

    Since MultiQuery dedupes in memory, we must apply the offset and limit in
    memory. The results that should be considered are
    results[lower_bound:upper_bound].

    We also pass the offset=0 and limit=upper_bound to the base queries to
    optimize performance.

    Args:
      config: The base datastore_query.QueryOptions.

    Returns:
      a tuple consisting of the lower_bound and upper_bound to impose in memory
      and the config to use with each bound query. The upper_bound may be None.
    """
    if config is None:
        return (0, None, None)
    lower_bound = config.offset or 0
    upper_bound = config.limit
    if lower_bound:
        if upper_bound is not None:
            upper_bound = min(lower_bound + upper_bound, _MAX_INT_32)
        config = datastore_query.QueryOptions(offset=0, limit=upper_bound, config=config)
    return (lower_bound, upper_bound, config)