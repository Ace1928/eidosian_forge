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
def _remove_pending(self, rpc):
    """Remove an RPC object from the list of pending RPCs.

    If the argument is a MultiRpc object, the wrapped RPCs are removed
    from the list of pending RPCs.
    """
    if isinstance(rpc, MultiRpc):
        for wrapped_rpc in rpc._MultiRpc__rpcs:
            self._remove_pending(wrapped_rpc)
    else:
        try:
            self.__pending_rpcs.remove(rpc)
        except KeyError:
            pass