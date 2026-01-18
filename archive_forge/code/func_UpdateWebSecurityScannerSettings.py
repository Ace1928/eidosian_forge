from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v1beta2 import securitycenter_v1beta2_messages as messages
def UpdateWebSecurityScannerSettings(self, request, global_params=None):
    """Update the WebSecurityScannerSettings resource.

      Args:
        request: (SecuritycenterProjectsUpdateWebSecurityScannerSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WebSecurityScannerSettings) The response message.
      """
    config = self.GetMethodConfig('UpdateWebSecurityScannerSettings')
    return self._RunMethod(config, request, global_params=global_params)