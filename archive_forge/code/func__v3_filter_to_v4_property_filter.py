from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def _v3_filter_to_v4_property_filter(self, v3_filter, v4_property_filter):
    """Converts a v3 Filter to a v4 PropertyFilter.

    Args:
      v3_filter: a datastore_pb.Filter
      v4_property_filter: a datastore_v4_pb.PropertyFilter to populate

    Raises:
      InvalidConversionError if the filter cannot be converted
    """
    check_conversion(v3_filter.property_size() == 1, 'invalid filter')
    check_conversion(v3_filter.op() <= 5, 'unsupported filter op: %d' % v3_filter.op())
    v4_property_filter.Clear()
    v4_property_filter.set_operator(v3_filter.op())
    v4_property_filter.mutable_property().set_name(v3_filter.property(0).name())
    self._entity_converter.v3_property_to_v4_value(v3_filter.property(0), True, v4_property_filter.mutable_value())