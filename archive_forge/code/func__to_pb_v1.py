from __future__ import absolute_import
from __future__ import unicode_literals
import base64
import collections
import pickle
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine._internal import six_subset
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_index
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.datastore import datastore_rpc
def _to_pb_v1(self, conn, query_options):
    """Returns a googledatastore.RunQueryRequest."""
    v1_req, v1_ancestor_filter = self._key_filter._to_pb_v1(conn.adapter)
    v1_query = v1_req.query
    if self.filter_predicate:
        filter_predicate_pb = self._filter_predicate._to_pb_v1(conn.adapter)
    if self.filter_predicate and v1_ancestor_filter:
        comp_filter_pb = v1_query.filter.composite_filter
        comp_filter_pb.op = googledatastore.CompositeFilter.AND
        comp_filter_pb.filters.add().CopyFrom(filter_predicate_pb)
        comp_filter_pb.filters.add().CopyFrom(v1_ancestor_filter)
    elif self.filter_predicate:
        v1_query.filter.CopyFrom(filter_predicate_pb)
    elif v1_ancestor_filter:
        v1_query.filter.CopyFrom(v1_ancestor_filter)
    if self._order:
        for order in self._order._to_pb_v1(conn.adapter):
            v1_query.order.add().CopyFrom(order)
    if QueryOptions.keys_only(query_options, conn.config):
        prop_ref_pb = v1_query.projection.add().property
        prop_ref_pb.name = datastore_pbs.PROPERTY_NAME_KEY
    projection = QueryOptions.projection(query_options, conn.config)
    self._validate_projection_and_group_by(projection, self._group_by)
    if projection:
        for prop in projection:
            prop_ref_pb = v1_query.projection.add().property
            prop_ref_pb.name = prop
    if self._group_by:
        for group_by in self._group_by:
            v1_query.distinct_on.add().name = group_by
    limit = QueryOptions.limit(query_options, conn.config)
    if limit is not None:
        v1_query.limit.value = limit
    count = QueryOptions.batch_size(query_options, conn.config)
    if count is None:
        count = QueryOptions.prefetch_size(query_options, conn.config)
    if count is not None:
        pass
    if query_options.offset:
        v1_query.offset = query_options.offset
    if query_options.start_cursor is not None:
        v1_query.start_cursor = query_options.start_cursor.to_bytes()
    if query_options.end_cursor is not None:
        v1_query.end_cursor = query_options.end_cursor.to_bytes()
    conn._set_request_read_policy(v1_req, query_options)
    conn._set_request_transaction(v1_req)
    return v1_req