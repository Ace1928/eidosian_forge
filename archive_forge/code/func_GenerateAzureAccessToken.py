from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkemulticloud.v1 import gkemulticloud_v1_messages as messages
def GenerateAzureAccessToken(self, request, global_params=None):
    """Generates a short-lived access token to authenticate to a given AzureCluster resource.

      Args:
        request: (GkemulticloudProjectsLocationsAzureClustersGenerateAzureAccessTokenRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGkemulticloudV1GenerateAzureAccessTokenResponse) The response message.
      """
    config = self.GetMethodConfig('GenerateAzureAccessToken')
    return self._RunMethod(config, request, global_params=global_params)