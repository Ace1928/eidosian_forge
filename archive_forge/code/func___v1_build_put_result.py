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
def __v1_build_put_result(self, v1_allocated_keys, user_data):
    """Internal method that builds the result of a put operation.

    Converts the results from a v1 AllocateIds operation to a list of user-level
    key objects.

    Args:
      v1_allocated_keys: a list of googledatastore.Keys that have been allocated
      user_data: a tuple consisting of:
        - a list of googledatastore.Entity objects
        - an optional extra_hook
    """
    v1_entities, extra_hook = user_data
    keys = []
    idx = 0
    for v1_entity in v1_entities:
        v1_entity = copy.deepcopy(v1_entity)
        if not datastore_pbs.is_complete_v1_key(v1_entity.key):
            v1_entity.key.CopyFrom(v1_allocated_keys[idx])
            idx += 1
        hashable_key = datastore_types.ReferenceToKeyValue(v1_entity.key)
        self.__pending_v1_deletes.pop(hashable_key, None)
        self.__pending_v1_upserts[hashable_key] = v1_entity
        keys.append(self.adapter.pb_v1_to_key(copy.deepcopy(v1_entity.key)))
    if extra_hook:
        keys = extra_hook(keys)
    return keys