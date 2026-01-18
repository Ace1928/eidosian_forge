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
def __query_result_hook(self, rpc):
    """Internal method used as get_result_hook for RunQuery/Next operation."""
    try:
        self._batch_shared.conn.check_rpc_success(rpc)
    except datastore_errors.NeedIndexError as exc:
        if isinstance(rpc.request, datastore_pb.Query):
            _, kind, ancestor, props = datastore_index.CompositeIndexForQuery(rpc.request)
            props = datastore_index.GetRecommendedIndexProperties(props)
            yaml = datastore_index.IndexYamlForQuery(kind, ancestor, props)
            xml = datastore_index.IndexXmlForQuery(kind, ancestor, props)
            raise datastore_errors.NeedIndexError('\n'.join([str(exc), self._need_index_header, yaml]), original_message=str(exc), header=self._need_index_header, yaml_index=yaml, xml_index=xml)
        raise
    query_result = rpc.response
    self._batch_shared.process_batch(query_result)
    if query_result.has_skipped_results_compiled_cursor():
        self.__skipped_cursor = Cursor(_cursor_bytes=query_result.skipped_results_compiled_cursor().Encode())
    self.__result_cursors = [Cursor(_cursor_bytes=result.Encode()) for result in query_result.result_compiled_cursor_list()]
    if query_result.has_compiled_cursor():
        self.__end_cursor = Cursor(_cursor_bytes=query_result.compiled_cursor().Encode())
    self._skipped_results = query_result.skipped_results()
    if query_result.more_results():
        self.__datastore_cursor = query_result.cursor()
        self.__more_results = True
    else:
        self._end()
    self.__results = self._process_results(query_result.result_list())
    return self