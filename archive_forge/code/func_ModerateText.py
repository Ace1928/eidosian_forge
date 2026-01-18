from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.language.v1 import language_v1_messages as messages
def ModerateText(self, request, global_params=None):
    """Moderates a document for harmful and sensitive categories.

      Args:
        request: (ModerateTextRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ModerateTextResponse) The response message.
      """
    config = self.GetMethodConfig('ModerateText')
    return self._RunMethod(config, request, global_params=global_params)