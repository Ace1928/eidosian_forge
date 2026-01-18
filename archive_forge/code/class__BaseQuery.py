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
class _BaseQuery(_BaseComponent):
    """A base class for query implementations."""

    def run(self, conn, query_options=None):
        """Runs the query using provided datastore_rpc.Connection.

    Args:
      conn: The datastore_rpc.Connection to use
      query_options: Optional query options to use

    Returns:
      A Batcher that implicitly fetches query results asynchronously.

    Raises:
      datastore_errors.BadArgumentError if any of the arguments are invalid.
    """
        return Batcher(query_options, self.run_async(conn, query_options))

    def run_async(self, conn, query_options=None):
        """Runs the query using the provided datastore_rpc.Connection.

    Args:
      conn: the datastore_rpc.Connection on which to run the query.
      query_options: Optional QueryOptions with which to run the query.

    Returns:
      An async object that can be used to grab the first Batch. Additional
      batches can be retrieved by calling Batch.next_batch/next_batch_async.

    Raises:
      datastore_errors.BadArgumentError if any of the arguments are invalid.
    """
        raise NotImplementedError

    def __getstate__(self):
        raise pickle.PicklingError('Pickling of %r is unsupported.' % self)