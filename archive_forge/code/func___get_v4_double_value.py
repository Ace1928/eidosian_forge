from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def __get_v4_double_value(self, v4_property):
    """Returns a double value from a v4 Property.

    Args:
      v4_property: an entity_v4_pb.Property

    Returns:
      a double

    Raises:
      InvalidConversionError: if the property doesn't contain a double value
    """
    check_conversion(v4_property.value().has_double_value(), 'Property does not contain a double value.')
    return v4_property.value().double_value()