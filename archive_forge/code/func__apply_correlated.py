from __future__ import absolute_import
from __future__ import unicode_literals
import base64
import collections
import pickle
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine._internal import six_subset
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_index
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.datastore import datastore_rpc
def _apply_correlated(self, value_maps):
    """Applies sub-filter to the correlated value maps.

    The default implementation matches when any value_map in value_maps
    matches the sub-filter.

    Args:
      value_maps: A list of correlated value_maps.
    Returns:
      True if any the entity matches the correlation filter.
    """
    for map in value_maps:
        if self._subfilter._apply(map):
            return True
    return False