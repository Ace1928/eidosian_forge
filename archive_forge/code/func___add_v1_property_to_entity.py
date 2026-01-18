from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def __add_v1_property_to_entity(self, v1_entity, v3_property, indexed):
    """Adds a v1 Property to an entity or modifies an existing one.

    Args:
      v1_entity: an googledatastore.Entity
      v3_property: an entity_pb.Property to convert to v1 and add to the dict
      indexed: whether the property is indexed
    """
    property_name = v3_property.name()
    v1_value = v1_entity.properties[property_name]
    if v3_property.multiple():
        self.v3_property_to_v1_value(v3_property, indexed, v1_value.array_value.values.add())
    else:
        self.v3_property_to_v1_value(v3_property, indexed, v1_value)