from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudkms.v1 import cloudkms_v1_messages as messages
def MacSign(self, request, global_params=None):
    """Signs data using a CryptoKeyVersion with CryptoKey.purpose MAC, producing a tag that can be verified by another source with the same key.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsMacSignRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (MacSignResponse) The response message.
      """
    config = self.GetMethodConfig('MacSign')
    return self._RunMethod(config, request, global_params=global_params)