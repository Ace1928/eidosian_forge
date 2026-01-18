from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.translate.v3beta1 import translate_v3beta1_messages as messages
def BatchTranslateText(self, request, global_params=None):
    """Translates a large volume of text in asynchronous batch mode. This function provides real-time output as the inputs are being processed. If caller cancels a request, the partial results (for an input file, it's all or nothing) may still be available on the specified output location. This call returns immediately and you can use google.longrunning.Operation.name to poll the status of the call.

      Args:
        request: (TranslateProjectsLocationsBatchTranslateTextRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('BatchTranslateText')
    return self._RunMethod(config, request, global_params=global_params)