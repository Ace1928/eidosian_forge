from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudkms.v1 import cloudkms_v1_messages as messages
def GetEkmConfig(self, request, global_params=None):
    """Returns the EkmConfig singleton resource for a given project and location.

      Args:
        request: (CloudkmsProjectsLocationsGetEkmConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (EkmConfig) The response message.
      """
    config = self.GetMethodConfig('GetEkmConfig')
    return self._RunMethod(config, request, global_params=global_params)