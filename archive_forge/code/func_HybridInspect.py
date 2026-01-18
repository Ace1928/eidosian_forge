from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dlp.v2 import dlp_v2_messages as messages
def HybridInspect(self, request, global_params=None):
    """Inspect hybrid content and store findings to a trigger. The inspection will be processed asynchronously. To review the findings monitor the jobs within the trigger.

      Args:
        request: (DlpProjectsLocationsJobTriggersHybridInspectRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2HybridInspectResponse) The response message.
      """
    config = self.GetMethodConfig('HybridInspect')
    return self._RunMethod(config, request, global_params=global_params)