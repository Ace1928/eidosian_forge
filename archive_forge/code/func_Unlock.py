from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.config.v1alpha2 import config_v1alpha2_messages as messages
def Unlock(self, request, global_params=None):
    """Unlocks a locked deployment.

      Args:
        request: (ConfigProjectsLocationsDeploymentsUnlockRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('Unlock')
    return self._RunMethod(config, request, global_params=global_params)