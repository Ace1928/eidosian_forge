from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkemulticloud.v1 import gkemulticloud_v1_messages as messages
def GetAzureServerConfig(self, request, global_params=None):
    """Returns information, such as supported Azure regions and Kubernetes versions, on a given Google Cloud location.

      Args:
        request: (GkemulticloudProjectsLocationsGetAzureServerConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGkemulticloudV1AzureServerConfig) The response message.
      """
    config = self.GetMethodConfig('GetAzureServerConfig')
    return self._RunMethod(config, request, global_params=global_params)