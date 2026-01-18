from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudkms.v1 import cloudkms_v1_messages as messages
def MacVerify(self, request, global_params=None):
    """Verifies MAC tag using a CryptoKeyVersion with CryptoKey.purpose MAC, and returns a response that indicates whether or not the verification was successful.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsMacVerifyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (MacVerifyResponse) The response message.
      """
    config = self.GetMethodConfig('MacVerify')
    return self._RunMethod(config, request, global_params=global_params)