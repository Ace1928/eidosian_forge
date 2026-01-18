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
class _BaseComponent(object):
    """A base class for query components.

  Currently just implements basic == and != functions.
  """

    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return NotImplemented
        return self is other or self.__dict__ == other.__dict__

    def __ne__(self, other):
        equal = self.__eq__(other)
        if equal is NotImplemented:
            return equal
        return not equal