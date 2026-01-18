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
def _to_pb(self, fetch_options=None):
    req = datastore_pb.NextRequest()
    if FetchOptions.produce_cursors(fetch_options, self._batch_shared.query_options, self._batch_shared.conn.config):
        req.set_compile(True)
    count = FetchOptions.batch_size(fetch_options, self._batch_shared.query_options, self._batch_shared.conn.config)
    if count is not None:
        req.set_count(count)
    if fetch_options is not None and fetch_options.offset:
        req.set_offset(fetch_options.offset)
    req.mutable_cursor().CopyFrom(self.__datastore_cursor)
    return req