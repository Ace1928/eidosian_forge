from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def _v3_filter_to_v1_property_filter(self, v3_filter, v1_property_filter):
    """Converts a v3 Filter to a v1 PropertyFilter.

    Args:
      v3_filter: a datastore_pb.Filter
      v1_property_filter: a googledatastore.PropertyFilter to populate

    Raises:
      InvalidConversionError if the filter cannot be converted
    """
    check_conversion(v3_filter.property_size() == 1, 'invalid filter')
    check_conversion(v3_filter.op() <= 5, 'unsupported filter op: %d' % v3_filter.op())
    v1_property_filter.Clear()
    v1_property_filter.op = v3_filter.op()
    v1_property_filter.property.name = v3_filter.property(0).name()
    self._entity_converter.v3_property_to_v1_value(v3_filter.property(0), True, v1_property_filter.value)