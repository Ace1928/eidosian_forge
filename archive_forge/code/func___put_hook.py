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
def __put_hook(self, rpc):
    """Internal method used as get_result_hook for Put operation."""
    self.check_rpc_success(rpc)
    entities_from_request, extra_hook = rpc.user_data
    if _CLOUD_DATASTORE_ENABLED and isinstance(rpc.response, googledatastore.CommitResponse):
        keys = []
        i = 0
        for entity in entities_from_request:
            if datastore_pbs.is_complete_v1_key(entity.key):
                keys.append(entity.key)
            else:
                keys.append(rpc.response.mutation_results[i].key)
                i += 1
        keys = [self.__adapter.pb_v1_to_key(key) for key in keys]
    else:
        keys = [self.__adapter.pb_to_key(key) for key in rpc.response.key_list()]
    if extra_hook is not None:
        keys = extra_hook(keys)
    return keys