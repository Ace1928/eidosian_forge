from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.telcoautomation.v1 import telcoautomation_v1_messages as messages
def ComputeDeploymentStatus(self, request, global_params=None):
    """Returns the requested deployment status.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsComputeDeploymentStatusRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ComputeDeploymentStatusResponse) The response message.
      """
    config = self.GetMethodConfig('ComputeDeploymentStatus')
    return self._RunMethod(config, request, global_params=global_params)