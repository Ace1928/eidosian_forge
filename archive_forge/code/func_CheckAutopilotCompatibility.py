from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.container.v1 import container_v1_messages as messages
def CheckAutopilotCompatibility(self, request, global_params=None):
    """Checks the cluster compatibility with Autopilot mode, and returns a list of compatibility issues.

      Args:
        request: (ContainerProjectsLocationsClustersCheckAutopilotCompatibilityRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CheckAutopilotCompatibilityResponse) The response message.
      """
    config = self.GetMethodConfig('CheckAutopilotCompatibility')
    return self._RunMethod(config, request, global_params=global_params)