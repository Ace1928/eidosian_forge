from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def __v4_double_property(self, name, value, indexed):
    """Creates a single-double-valued v4 Property.

    Args:
      name: the property name
      value: the double value of the property
      indexed: whether the value should be indexed

    Returns:
      an entity_v4_pb.Property
    """
    v4_property = entity_v4_pb.Property()
    v4_property.set_name(name)
    v4_value = v4_property.mutable_value()
    v4_value.set_indexed(indexed)
    v4_value.set_double_value(value)
    return v4_property