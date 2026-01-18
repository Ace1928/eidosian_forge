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
def _generate_pb_lists(self, grouped_values, base_size, max_count, max_groups, config):
    """Internal helper: repeatedly yield a list of 2 elements.

    Args:
      grouped_values: A list of lists.  The inner lists consist of objects
        grouped by e.g. entity group or id sequence.

      base_size: An integer representing the base size of an rpc.  Used for
        splitting operations across multiple RPCs due to size limitations.

      max_count: An integer representing the maximum number of objects we can
        send in an rpc.  Used for splitting operations across multiple RPCs.

      max_groups: An integer representing the maximum number of groups we can
        have represented in an rpc.  Can be None, in which case no constraint.

      config: The config object, defining max rpc size in bytes.

    Yields:
      Repeatedly yields 2 element tuples.  The first element is a list of
      protobufs to send in one batch.  The second element is a list containing
      the original location of those protobufs (expressed as an index) in the
      input.
    """
    max_size = Configuration.max_rpc_bytes(config, self.__config) or self.MAX_RPC_BYTES
    pbs = []
    pb_indexes = []
    size = base_size
    num_groups = 0
    for indexed_pbs in grouped_values:
        num_groups += 1
        if max_groups is not None and num_groups > max_groups:
            yield (pbs, pb_indexes)
            pbs = []
            pb_indexes = []
            size = base_size
            num_groups = 1
        for indexed_pb in indexed_pbs:
            pb, index = indexed_pb
            incr_size = pb.ByteSize() + 5
            if not isinstance(config, apiproxy_stub_map.UserRPC) and (len(pbs) >= max_count or (pbs and size + incr_size > max_size)):
                yield (pbs, pb_indexes)
                pbs = []
                pb_indexes = []
                size = base_size
                num_groups = 1
            pbs.append(pb)
            pb_indexes.append(index)
            size += incr_size
    yield (pbs, pb_indexes)