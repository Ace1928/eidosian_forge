from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dlp.v2 import dlp_v2_messages as messages
def Inspect(self, request, global_params=None):
    """Finds potentially sensitive info in content. This method has limits on input size, processing time, and output size. When no InfoTypes or CustomInfoTypes are specified in this request, the system will automatically choose what detectors to run. By default this may be all types, but may change over time as detectors are updated. For how to guides, see https://cloud.google.com/sensitive-data-protection/docs/inspecting-images and https://cloud.google.com/sensitive-data-protection/docs/inspecting-text,.

      Args:
        request: (DlpProjectsLocationsContentInspectRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2InspectContentResponse) The response message.
      """
    config = self.GetMethodConfig('Inspect')
    return self._RunMethod(config, request, global_params=global_params)