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
class _IgnoreFilter(_SinglePropertyFilter):
    """A filter that removes all entities with the given keys."""

    def __init__(self, key_value_set):
        super(_IgnoreFilter, self).__init__()
        self._keys = key_value_set

    def _get_prop_name(self):
        return datastore_types.KEY_SPECIAL_PROPERTY

    def _apply_to_value(self, value):
        return value not in self._keys