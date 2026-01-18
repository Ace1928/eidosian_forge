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
def Hint(self, hint):
    """Sets a hint for how this query should run.

    The query hint gives us information about how best to execute your query.
    Currently, we can only do one index scan, so the query hint should be used
    to indicates which index we should scan against.

    Use FILTER_FIRST if your first filter will only match a few results. In
    this case, it will be most efficient to scan against the index for this
    property, load the results into memory, and apply the remaining filters
    and sort orders there.

    Similarly, use ANCESTOR_FIRST if the query's ancestor only has a few
    descendants. In this case, it will be most efficient to scan all entities
    below the ancestor and load them into memory first.

    Use ORDER_FIRST if the query has a sort order and the result set is large
    or you only plan to fetch the first few results. In that case, we
    shouldn't try to load all of the results into memory; instead, we should
    scan the index for this property, which is in sorted order.

    Note that hints are currently ignored in the v3 datastore!

    Arg:
      one of datastore.Query.[ORDER_FIRST, ANCESTOR_FIRST, FILTER_FIRST]

    Returns:
      # this query
      Query
    """
    if hint is not self.__query_options.hint:
        self.__query_options = datastore_query.QueryOptions(hint=hint, config=self.__query_options)
    return self