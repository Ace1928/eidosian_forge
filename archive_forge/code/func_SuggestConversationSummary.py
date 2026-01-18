from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
def SuggestConversationSummary(self, request, global_params=None):
    """Suggests summary for a conversation based on specific historical messages. The range of the messages to be used for summary can be specified in the request.

      Args:
        request: (DialogflowProjectsLocationsConversationsSuggestionsSuggestConversationSummaryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2SuggestConversationSummaryResponse) The response message.
      """
    config = self.GetMethodConfig('SuggestConversationSummary')
    return self._RunMethod(config, request, global_params=global_params)