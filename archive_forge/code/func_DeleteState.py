from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.config.v1alpha2 import config_v1alpha2_messages as messages
def DeleteState(self, request, global_params=None):
    """Deletes Terraform state file in a given deployment.

      Args:
        request: (ConfigProjectsLocationsDeploymentsDeleteStateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
    config = self.GetMethodConfig('DeleteState')
    return self._RunMethod(config, request, global_params=global_params)