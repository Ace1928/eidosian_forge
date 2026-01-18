from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
def SearchKnowledge(self, request, global_params=None):
    """Get answers for the given query based on knowledge documents.

      Args:
        request: (GoogleCloudDialogflowV2SearchKnowledgeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2SearchKnowledgeResponse) The response message.
      """
    config = self.GetMethodConfig('SearchKnowledge')
    return self._RunMethod(config, request, global_params=global_params)