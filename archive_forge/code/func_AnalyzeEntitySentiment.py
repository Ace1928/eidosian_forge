from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.language.v1 import language_v1_messages as messages
def AnalyzeEntitySentiment(self, request, global_params=None):
    """Finds entities, similar to AnalyzeEntities in the text and analyzes sentiment associated with each entity and its mentions.

      Args:
        request: (AnalyzeEntitySentimentRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AnalyzeEntitySentimentResponse) The response message.
      """
    config = self.GetMethodConfig('AnalyzeEntitySentiment')
    return self._RunMethod(config, request, global_params=global_params)