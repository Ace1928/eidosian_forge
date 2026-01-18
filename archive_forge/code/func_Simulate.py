from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v1 import securitycenter_v1_messages as messages
def Simulate(self, request, global_params=None):
    """Simulates a given SecurityHealthAnalyticsCustomModule and Resource.

      Args:
        request: (SecuritycenterProjectsSecurityHealthAnalyticsSettingsCustomModulesSimulateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SimulateSecurityHealthAnalyticsCustomModuleResponse) The response message.
      """
    config = self.GetMethodConfig('Simulate')
    return self._RunMethod(config, request, global_params=global_params)