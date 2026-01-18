from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudkms.v1 import cloudkms_v1_messages as messages
def VerifyConnectivity(self, request, global_params=None):
    """Verifies that Cloud KMS can successfully connect to the external key manager specified by an EkmConnection. If there is an error connecting to the EKM, this method returns a FAILED_PRECONDITION status containing structured information as described at https://cloud.google.com/kms/docs/reference/ekm_errors.

      Args:
        request: (CloudkmsProjectsLocationsEkmConnectionsVerifyConnectivityRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (VerifyConnectivityResponse) The response message.
      """
    config = self.GetMethodConfig('VerifyConnectivity')
    return self._RunMethod(config, request, global_params=global_params)