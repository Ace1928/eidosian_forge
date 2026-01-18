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
class _AugmentedQuery(_BaseQuery):
    """A query that combines a datastore query with in-memory filters/results."""

    @datastore_rpc._positional(2)
    def __init__(self, query, in_memory_results=None, in_memory_filter=None, max_filtered_count=None):
        """Constructor for _AugmentedQuery.

    Do not call directly. Use the utility functions instead (e.g.
    datastore_query.inject_results)

    Args:
      query: A datastore_query.Query object to augment.
      in_memory_results: a list of pre- sorted and filtered result to add to the
        stream of datastore results or None .
      in_memory_filter: a set of in-memory filters to apply to the datastore
        results or None.
      max_filtered_count: the maximum number of datastore entities that will be
        filtered out by in_memory_filter if known.
    """
        if not isinstance(query, Query):
            raise datastore_errors.BadArgumentError('query argument should be datastore_query.Query (%r)' % (query,))
        if in_memory_filter is not None and (not isinstance(in_memory_filter, FilterPredicate)):
            raise datastore_errors.BadArgumentError('in_memory_filter argument should be ' + 'datastore_query.FilterPredicate (%r)' % (in_memory_filter,))
        if in_memory_results is not None and (not isinstance(in_memory_results, list)):
            raise datastore_errors.BadArgumentError('in_memory_results argument should be a list of' + 'datastore_pv.EntityProto (%r)' % (in_memory_results,))
        datastore_types.ValidateInteger(max_filtered_count, 'max_filtered_count', empty_ok=True, zero_ok=True)
        self._query = query
        self._max_filtered_count = max_filtered_count
        self._in_memory_filter = in_memory_filter
        self._in_memory_results = in_memory_results

    @property
    def app(self):
        return self._query._key_filter.app

    @property
    def namespace(self):
        return self._query._key_filter.namespace

    @property
    def kind(self):
        return self._query._key_filter.kind

    @property
    def ancestor(self):
        return self._query._key_filter.ancestor

    @property
    def filter_predicate(self):
        return self._query._filter_predicate

    @property
    def order(self):
        return self._query._order

    @property
    def group_by(self):
        return self._query._group_by

    def run_async(self, conn, query_options=None):
        if not isinstance(conn, datastore_rpc.BaseConnection):
            raise datastore_errors.BadArgumentError('conn should be a datastore_rpc.BaseConnection (%r)' % (conn,))
        if not QueryOptions.is_configuration(query_options):
            query_options = QueryOptions(config=query_options)
        if self._query._order:
            changes = {'keys_only': False}
        else:
            changes = {}
        if self._in_memory_filter or self._in_memory_results:
            in_memory_offset = query_options.offset
            in_memory_limit = query_options.limit
            if in_memory_limit is not None:
                if self._in_memory_filter is None:
                    changes['limit'] = in_memory_limit
                elif self._max_filtered_count is not None:
                    changes['limit'] = in_memory_limit + self._max_filtered_count
                else:
                    changes['limit'] = None
            if in_memory_offset:
                changes['offset'] = None
                if changes.get('limit', None) is not None:
                    changes['limit'] += in_memory_offset
            else:
                in_memory_offset = None
        else:
            in_memory_offset = None
            in_memory_limit = None
        modified_query_options = QueryOptions(config=query_options, **changes)
        if conn._api_version == datastore_rpc._CLOUD_DATASTORE_V1:
            req = self._query._to_pb_v1(conn, modified_query_options)
        else:
            req = self._query._to_pb(conn, modified_query_options)
        start_cursor = query_options.start_cursor
        if not start_cursor and query_options.produce_cursors:
            start_cursor = Cursor()
        return _AugmentedBatch.create_async(self, modified_query_options, conn, req, in_memory_offset=in_memory_offset, in_memory_limit=in_memory_limit, start_cursor=start_cursor)