from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.documentai.v1 import documentai_v1_messages as messages
def SetDefaultProcessorVersion(self, request, global_params=None):
    """Set the default (active) version of a Processor that will be used in ProcessDocument and BatchProcessDocuments.

      Args:
        request: (DocumentaiProjectsLocationsProcessorsSetDefaultProcessorVersionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
    config = self.GetMethodConfig('SetDefaultProcessorVersion')
    return self._RunMethod(config, request, global_params=global_params)