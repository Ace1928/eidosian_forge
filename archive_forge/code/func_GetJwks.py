from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkemulticloud.v1 import gkemulticloud_v1_messages as messages
def GetJwks(self, request, global_params=None):
    """Gets the public component of the cluster signing keys in JSON Web Key format.

      Args:
        request: (GkemulticloudProjectsLocationsAzureClustersGetJwksRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGkemulticloudV1AzureJsonWebKeys) The response message.
      """
    config = self.GetMethodConfig('GetJwks')
    return self._RunMethod(config, request, global_params=global_params)