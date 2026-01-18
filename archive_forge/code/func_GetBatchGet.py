from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.source.v1 import source_v1_messages as messages
def GetBatchGet(self, request, global_params=None):
    """Retrieves revision metadata for several revisions at once. It returns an.
error if any retrieval fails.

      Args:
        request: (SourceProjectsReposRevisionsGetBatchGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GetRevisionsResponse) The response message.
      """
    config = self.GetMethodConfig('GetBatchGet')
    return self._RunMethod(config, request, global_params=global_params)