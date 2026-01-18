from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.firestore.v1 import firestore_v1_messages as messages
def PartitionQuery(self, request, global_params=None):
    """Partitions a query by returning partition cursors that can be used to run the query in parallel. The returned partition cursors are split points that can be used by RunQuery as starting/end points for the query results.

      Args:
        request: (FirestoreProjectsDatabasesDocumentsPartitionQueryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PartitionQueryResponse) The response message.
      """
    config = self.GetMethodConfig('PartitionQuery')
    return self._RunMethod(config, request, global_params=global_params)