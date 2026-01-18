from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def __add_v3_property_from_v4(self, property_name, is_multi, is_projection, v4_value, v3_entity):
    """Adds a v3 Property to an Entity based on information from a v4 Property.

    Args:
      property_name: the name of the property
      is_multi: whether the property contains multiple values
      is_projection: whether the property is a projection
      v4_value: an entity_v4_pb.Value
      v3_entity: an entity_pb.EntityProto
    """
    if v4_value.indexed():
        self.v4_to_v3_property(property_name, is_multi, is_projection, v4_value, v3_entity.add_property())
    else:
        self.v4_to_v3_property(property_name, is_multi, is_projection, v4_value, v3_entity.add_raw_property())