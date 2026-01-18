from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.storage.v1 import storage_v1_messages as messages
def BulkRestore(self, request, global_params=None):
    """Initiates a long-running bulk restore operation on the specified bucket.

      Args:
        request: (StorageObjectsBulkRestoreRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
    config = self.GetMethodConfig('BulkRestore')
    return self._RunMethod(config, request, global_params=global_params)