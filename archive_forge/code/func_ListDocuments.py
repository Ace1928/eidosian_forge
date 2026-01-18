from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.firestore.v1 import firestore_v1_messages as messages
def ListDocuments(self, request, global_params=None):
    """Lists documents.

      Args:
        request: (FirestoreProjectsDatabasesDocumentsListDocumentsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDocumentsResponse) The response message.
      """
    config = self.GetMethodConfig('ListDocuments')
    return self._RunMethod(config, request, global_params=global_params)