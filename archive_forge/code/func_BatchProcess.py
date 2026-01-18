from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.documentai.v1 import documentai_v1_messages as messages
def BatchProcess(self, request, global_params=None):
    """LRO endpoint to batch process many documents. The output is written to Cloud Storage as JSON in the [Document] format.

      Args:
        request: (DocumentaiProjectsLocationsProcessorsBatchProcessRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
    config = self.GetMethodConfig('BatchProcess')
    return self._RunMethod(config, request, global_params=global_params)