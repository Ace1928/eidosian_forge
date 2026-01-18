from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
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