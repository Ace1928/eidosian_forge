from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.documentai.v1 import documentai_v1_messages as messages
def FetchProcessorTypes(self, request, global_params=None):
    """Fetches processor types. Note that we don't use ListProcessorTypes here, because it isn't paginated.

      Args:
        request: (DocumentaiProjectsLocationsFetchProcessorTypesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDocumentaiV1FetchProcessorTypesResponse) The response message.
      """
    config = self.GetMethodConfig('FetchProcessorTypes')
    return self._RunMethod(config, request, global_params=global_params)