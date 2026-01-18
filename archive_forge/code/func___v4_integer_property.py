from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def __v4_integer_property(self, name, value, indexed):
    """Creates a single-integer-valued v4 Property.

    Args:
      name: the property name
      value: the integer value of the property
      indexed: whether the value should be indexed

    Returns:
      an entity_v4_pb.Property
    """
    v4_property = entity_v4_pb.Property()
    v4_property.set_name(name)
    v4_value = v4_property.mutable_value()
    v4_value.set_indexed(indexed)
    v4_value.set_integer_value(value)
    return v4_property