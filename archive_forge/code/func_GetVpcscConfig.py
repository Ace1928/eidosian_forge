from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.artifactregistry.v1 import artifactregistry_v1_messages as messages
def GetVpcscConfig(self, request, global_params=None):
    """Retrieves the VPCSC Config for the Project.

      Args:
        request: (ArtifactregistryProjectsLocationsGetVpcscConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (VPCSCConfig) The response message.
      """
    config = self.GetMethodConfig('GetVpcscConfig')
    return self._RunMethod(config, request, global_params=global_params)