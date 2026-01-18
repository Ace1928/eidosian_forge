from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.documentai.v1 import documentai_v1_messages as messages
def EvaluateProcessorVersion(self, request, global_params=None):
    """Evaluates a ProcessorVersion against annotated documents, producing an Evaluation.

      Args:
        request: (DocumentaiProjectsLocationsProcessorsProcessorVersionsEvaluateProcessorVersionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
    config = self.GetMethodConfig('EvaluateProcessorVersion')
    return self._RunMethod(config, request, global_params=global_params)