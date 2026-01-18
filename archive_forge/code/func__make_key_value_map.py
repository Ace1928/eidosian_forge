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
def _make_key_value_map(entity, property_names):
    """Extracts key values from the given entity.

  Args:
    entity: The entity_pb.EntityProto to extract values from.
    property_names: The names of the properties from which to extract values.

  Returns:
    A dict mapping property names to a lists of key values.
  """
    value_map = dict(((name, []) for name in property_names))
    for prop in entity.property_list():
        if prop.name() in value_map:
            value_map[prop.name()].append(datastore_types.PropertyValueToKeyValue(prop.value()))
    if datastore_types.KEY_SPECIAL_PROPERTY in value_map:
        value_map[datastore_types.KEY_SPECIAL_PROPERTY] = [datastore_types.ReferenceToKeyValue(entity.key())]
    return value_map