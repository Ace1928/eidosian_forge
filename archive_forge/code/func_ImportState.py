from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.config.v1alpha2 import config_v1alpha2_messages as messages
def ImportState(self, request, global_params=None):
    """Imports Terraform state file in a given deployment. The state file does not take effect until the Deployment has been unlocked.

      Args:
        request: (ConfigProjectsLocationsDeploymentsImportStateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Statefile) The response message.
      """
    config = self.GetMethodConfig('ImportState')
    return self._RunMethod(config, request, global_params=global_params)