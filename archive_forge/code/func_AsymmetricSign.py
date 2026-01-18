from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudkms.v1 import cloudkms_v1_messages as messages
def AsymmetricSign(self, request, global_params=None):
    """Signs data using a CryptoKeyVersion with CryptoKey.purpose ASYMMETRIC_SIGN, producing a signature that can be verified with the public key retrieved from GetPublicKey.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsAsymmetricSignRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AsymmetricSignResponse) The response message.
      """
    config = self.GetMethodConfig('AsymmetricSign')
    return self._RunMethod(config, request, global_params=global_params)