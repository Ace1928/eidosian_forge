from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def __v4_to_v3_point_value(self, v4_point_entity, v3_point_value):
    """Converts a v4 point Entity to a v3 PointValue.

    Args:
      v4_point_entity: an entity_v4_pb.Entity representing a point
      v3_point_value: an entity_pb.Property_PointValue to populate
    """
    v3_point_value.Clear()
    name_to_v4_property = self.__build_name_to_v4_property_map(v4_point_entity)
    v3_point_value.set_x(self.__get_v4_double_value(name_to_v4_property['x']))
    v3_point_value.set_y(self.__get_v4_double_value(name_to_v4_property['y']))