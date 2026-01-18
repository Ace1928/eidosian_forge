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
class _BatchShared(object):
    """Data shared among the batches of a query."""

    def __init__(self, query, query_options, conn, augmented_query=None, initial_offset=None):
        self.__query = query
        self.__query_options = query_options
        self.__conn = conn
        self.__augmented_query = augmented_query
        self.__was_first_result_processed = False
        if initial_offset is None:
            initial_offset = query_options.offset or 0
        self.__expected_offset = initial_offset
        self.__remaining_limit = query_options.limit

    @property
    def query(self):
        return self.__query

    @property
    def query_options(self):
        return self.__query_options

    @property
    def conn(self):
        return self.__conn

    @property
    def augmented_query(self):
        return self.__augmented_query

    @property
    def keys_only(self):
        return self.__keys_only

    @property
    def compiled_query(self):
        return self.__compiled_query

    @property
    def expected_offset(self):
        return self.__expected_offset

    @property
    def remaining_limit(self):
        return self.__remaining_limit

    @property
    def index_list(self):
        """Returns the list of indexes used by the query.
    Possibly None when the adapter does not implement pb_to_index.
    """
        return self.__index_list

    def process_batch(self, batch):
        if self.conn._api_version == datastore_rpc._CLOUD_DATASTORE_V1:
            skipped_results = batch.skipped_results
            num_results = len(batch.entity_results)
        else:
            skipped_results = batch.skipped_results()
            num_results = batch.result_size()
        self.__expected_offset -= skipped_results
        if self.__remaining_limit is not None:
            self.__remaining_limit -= num_results
        if not self.__was_first_result_processed:
            self.__was_first_result_processed = True
            if self.conn._api_version == datastore_rpc._CLOUD_DATASTORE_V1:
                result_type = batch.entity_result_type
                self.__keys_only = result_type == googledatastore.EntityResult.KEY_ONLY
                self.__compiled_query = None
                self.__index_list = None
            else:
                self.__keys_only = batch.keys_only()
                if batch.has_compiled_query():
                    self.__compiled_query = batch.compiled_query
                else:
                    self.__compiled_query = None
                try:
                    self.__index_list = [self.__conn.adapter.pb_to_index(index_pb) for index_pb in batch.index_list()]
                except NotImplementedError:
                    self.__index_list = None