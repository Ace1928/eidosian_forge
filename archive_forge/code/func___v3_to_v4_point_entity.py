from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def __v3_to_v4_point_entity(self, v3_point_value, v4_entity):
    """Converts a v3 UserValue to a v4 user Entity.

    Args:
      v3_point_value: an entity_pb.Property_PointValue
      v4_entity: an entity_v4_pb.Entity to populate
    """
    v4_entity.Clear()
    v4_entity.property_list().append(self.__v4_double_property(PROPERTY_NAME_X, v3_point_value.x(), False))
    v4_entity.property_list().append(self.__v4_double_property(PROPERTY_NAME_Y, v3_point_value.y(), False))