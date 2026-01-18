from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkemulticloud.v1 import gkemulticloud_v1_messages as messages
def GetAwsServerConfig(self, request, global_params=None):
    """Returns information, such as supported AWS regions and Kubernetes versions, on a given Google Cloud location.

      Args:
        request: (GkemulticloudProjectsLocationsGetAwsServerConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGkemulticloudV1AwsServerConfig) The response message.
      """
    config = self.GetMethodConfig('GetAwsServerConfig')
    return self._RunMethod(config, request, global_params=global_params)