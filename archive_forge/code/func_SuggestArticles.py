from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
def SuggestArticles(self, request, global_params=None):
    """Gets suggested articles for a participant based on specific historical messages.

      Args:
        request: (DialogflowProjectsLocationsConversationsParticipantsSuggestionsSuggestArticlesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2SuggestArticlesResponse) The response message.
      """
    config = self.GetMethodConfig('SuggestArticles')
    return self._RunMethod(config, request, global_params=global_params)