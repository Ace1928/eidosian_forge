from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
class _QueryConverter(object):
    """Base converter for v3 and v1 queries."""

    def __init__(self, entity_converter):
        self._entity_converter = entity_converter

    def get_entity_converter(self):
        return self._entity_converter

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

    def _v3_query_to_v1_ancestor_filter(self, v3_query, v1_property_filter):
        """Converts a v3 Query to a v1 ancestor PropertyFilter.

    Args:
      v3_query: a datastore_pb.Query
      v1_property_filter: a googledatastore.PropertyFilter to populate
    """
        v1_property_filter.Clear()
        v1_property_filter.set_operator(v3_query.shallow() and googledatastore.PropertyFilter.HAS_PARENT or googledatastore.PropertyFilter.HAS_ANCESTOR)
        prop = v1_property_filter.property
        prop.set_name(PROPERTY_NAME_KEY)
        if v3_query.has_ancestor():
            self._entity_converter.v3_to_v1_key(v3_query.ancestor(), v1_property_filter.value.mutable_key_value)
        else:
            v1_property_filter.value.null_value = googledatastore.NULL_VALUE

    def v3_order_to_v1_order(self, v3_order, v1_order):
        """Converts a v3 Query order to a v1 PropertyOrder.

    Args:
      v3_order: a datastore_pb.Query.Order
      v1_order: a googledatastore.PropertyOrder to populate
    """
        v1_order.property.name = v3_order.property()
        if v3_order.has_direction():
            v1_order.direction = v3_order.direction()

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

    def _v3_query_to_v4_ancestor_filter(self, v3_query, v4_property_filter):
        """Converts a v3 Query to a v4 ancestor PropertyFilter.

    Args:
      v3_query: a datastore_pb.Query
      v4_property_filter: a datastore_v4_pb.PropertyFilter to populate
    """
        v4_property_filter.Clear()
        v4_property_filter.set_operator(datastore_v4_pb.PropertyFilter.HAS_ANCESTOR)
        prop = v4_property_filter.mutable_property()
        prop.set_name(PROPERTY_NAME_KEY)
        self._entity_converter.v3_to_v4_key(v3_query.ancestor(), v4_property_filter.mutable_value().mutable_key_value())

    def v3_order_to_v4_order(self, v3_order, v4_order):
        """Converts a v3 Query order to a v4 PropertyOrder.

    Args:
      v3_order: a datastore_pb.Query.Order
      v4_order: a datastore_v4_pb.PropertyOrder to populate
    """
        v4_order.mutable_property().set_name(v3_order.property())
        if v3_order.has_direction():
            v4_order.set_direction(v3_order.direction())