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
def __create_result_index_pairs(self, indexes):
    """Internal helper: build a function that ties an index with each result.

    Args:
      indexes: A list of integers.  A value x at location y in the list means
        that the result at location y in the result list needs to be at location
        x in the list of results returned to the user.
    """

    def create_result_index_pairs(results):
        return list(zip(results, indexes))
    return create_result_index_pairs