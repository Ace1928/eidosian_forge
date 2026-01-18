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
class _ReverseOrder(_BaseComponent):
    """Reverses the comparison for the given object."""

    def __init__(self, obj):
        """Constructor for _ReverseOrder.

    Args:
      obj: Any comparable and hashable object.
    """
        super(_ReverseOrder, self).__init__()
        self._obj = obj

    def __hash__(self):
        return hash(self._obj)

    def __cmp__(self, other):
        assert self.__class__ == other.__class__, 'A datastore_query._ReverseOrder object can only be compared to an object of the same type.'
        return -cmp(self._obj, other._obj)