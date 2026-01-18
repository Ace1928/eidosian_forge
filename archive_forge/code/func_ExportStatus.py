from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.run.v2 import run_v2_messages as messages
def ExportStatus(self, request, global_params=None):
    """Read the status of an image export operation.

      Args:
        request: (RunProjectsLocationsServicesRevisionsExportStatusRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRunV2ExportStatusResponse) The response message.
      """
    config = self.GetMethodConfig('ExportStatus')
    return self._RunMethod(config, request, global_params=global_params)