from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.firestore.v1 import firestore_v1_messages as messages
def Commit(self, request, global_params=None):
    """Commits a transaction, while optionally updating documents.

      Args:
        request: (FirestoreProjectsDatabasesDocumentsCommitRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CommitResponse) The response message.
      """
    config = self.GetMethodConfig('Commit')
    return self._RunMethod(config, request, global_params=global_params)