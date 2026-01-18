from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.language.v1 import language_v1_messages as messages
def AnalyzeSentiment(self, request, global_params=None):
    """Analyzes the sentiment of the provided text.

      Args:
        request: (AnalyzeSentimentRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AnalyzeSentimentResponse) The response message.
      """
    config = self.GetMethodConfig('AnalyzeSentiment')
    return self._RunMethod(config, request, global_params=global_params)