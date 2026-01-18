from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
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