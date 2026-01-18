from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1beta1 import healthcare_v1beta1_messages as messages
def SetBlobStorageSettings(self, request, global_params=None):
    """SetBlobStorageSettings sets the blob storage settings of the specified resources.

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresSetBlobStorageSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('SetBlobStorageSettings')
    return self._RunMethod(config, request, global_params=global_params)