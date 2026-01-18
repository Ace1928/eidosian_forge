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
def Count(self, limit=1000, **kwargs):
    """Return the number of matched entities for this query.

    Will return the de-duplicated count of results.  Will call the more
    efficient Get() function if a limit is given.

    Args:
      limit: maximum number of entries to count (for any result > limit, return
      limit).
      config: Optional Configuration to use for this request.

    Returns:
      count of the number of entries returned.
    """
    kwargs['limit'] = limit
    config = _GetConfigFromKwargs(kwargs, convert_rpc=True, config_class=datastore_query.QueryOptions)
    projection, override = self.__GetProjectionOverride(config)
    if not projection:
        config = datastore_query.QueryOptions(keys_only=True, config=config)
    elif override:
        config = datastore_query.QueryOptions(projection=override, config=config)
    lower_bound, upper_bound, config = self._ExtractBounds(config)
    used_keys = set()
    for bound_query in self.__bound_queries:
        for result in bound_query.Run(config=config):
            if projection:
                dedupe_key = (result.key(), tuple(result.iteritems()))
            else:
                dedupe_key = result
            used_keys.add(dedupe_key)
            if upper_bound and len(used_keys) >= upper_bound:
                return upper_bound - lower_bound
    return max(0, len(used_keys) - lower_bound)