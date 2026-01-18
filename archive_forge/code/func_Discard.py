from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.telcoautomation.v1 import telcoautomation_v1_messages as messages
def Discard(self, request, global_params=None):
    """Discards the changes in a deployment and reverts the deployment to the last approved deployment revision. No changes take place if a deployment does not have revisions.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsDiscardRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DiscardDeploymentChangesResponse) The response message.
      """
    config = self.GetMethodConfig('Discard')
    return self._RunMethod(config, request, global_params=global_params)