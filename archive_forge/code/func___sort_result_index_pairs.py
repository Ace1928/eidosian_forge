from __future__ import absolute_import
from __future__ import unicode_literals
import collections
import copy
import functools
import logging
import os
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine._internal import six_subset
from googlecloudsdk.third_party.appengine.api import api_base_pb
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.api import apiproxy_stub_map
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
def __sort_result_index_pairs(self, extra_hook):
    """Builds a function that sorts the indexed results.

    Args:
      extra_hook: A function that the returned function will apply to its result
        before returning.

    Returns:
      A function that takes a list of results and reorders them to match the
      order in which the input values associated with each results were
      originally provided.
    """

    def sort_result_index_pairs(result_index_pairs):
        results = [None] * len(result_index_pairs)
        for result, index in result_index_pairs:
            results[index] = result
        if extra_hook is not None:
            results = extra_hook(results)
        return results
    return sort_result_index_pairs