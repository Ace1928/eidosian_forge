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
def _map_and_group(self, values, map_fn, group_fn):
    """Internal helper: map values to keys and group by key. Here key is any
    object derived from an input value by map_fn, and which can be grouped
    by group_fn.

    Args:
      values: The values to be grouped by applying get_group(to_ref(value)).
      map_fn: a function that maps a value to a key to be grouped.
      group_fn: a function that groups the keys output by map_fn.

    Returns:
      A list where each element is a list of (key, index) pairs.  Here
      index is the location of the value from which the key was derived in
      the original list.
    """
    indexed_key_groups = collections.defaultdict(list)
    for index, value in enumerate(values):
        key = map_fn(value)
        indexed_key_groups[group_fn(key)].append((key, index))
    return list(indexed_key_groups.values())