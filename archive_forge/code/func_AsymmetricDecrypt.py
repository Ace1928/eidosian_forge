from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudkms.v1 import cloudkms_v1_messages as messages
def AsymmetricDecrypt(self, request, global_params=None):
    """Decrypts data that was encrypted with a public key retrieved from GetPublicKey corresponding to a CryptoKeyVersion with CryptoKey.purpose ASYMMETRIC_DECRYPT.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsAsymmetricDecryptRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AsymmetricDecryptResponse) The response message.
      """
    config = self.GetMethodConfig('AsymmetricDecrypt')
    return self._RunMethod(config, request, global_params=global_params)