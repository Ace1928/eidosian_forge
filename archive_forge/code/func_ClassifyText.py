from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.language.v1 import language_v1_messages as messages
def ClassifyText(self, request, global_params=None):
    """Classifies a document into categories.

      Args:
        request: (ClassifyTextRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ClassifyTextResponse) The response message.
      """
    config = self.GetMethodConfig('ClassifyText')
    return self._RunMethod(config, request, global_params=global_params)