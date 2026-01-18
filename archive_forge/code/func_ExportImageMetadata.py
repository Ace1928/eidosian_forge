from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.run.v2 import run_v2_messages as messages
def ExportImageMetadata(self, request, global_params=None):
    """Export image metadata for a given resource.

      Args:
        request: (RunProjectsLocationsExportImageMetadataRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRunV2Metadata) The response message.
      """
    config = self.GetMethodConfig('ExportImageMetadata')
    return self._RunMethod(config, request, global_params=global_params)