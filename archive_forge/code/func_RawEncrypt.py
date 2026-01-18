from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudkms.v1 import cloudkms_v1_messages as messages
def RawEncrypt(self, request, global_params=None):
    """Encrypts data using portable cryptographic primitives. Most users should choose Encrypt and Decrypt rather than their raw counterparts. The CryptoKey.purpose must be RAW_ENCRYPT_DECRYPT.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsRawEncryptRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RawEncryptResponse) The response message.
      """
    config = self.GetMethodConfig('RawEncrypt')
    return self._RunMethod(config, request, global_params=global_params)