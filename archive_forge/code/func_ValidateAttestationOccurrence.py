from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.binaryauthorization.v1alpha2 import binaryauthorization_v1alpha2_messages as messages
def ValidateAttestationOccurrence(self, request, global_params=None):
    """Returns whether the given `Attestation` for the given image URI was signed by the given `Attestor`.

      Args:
        request: (BinaryauthorizationProjectsAttestorsValidateAttestationOccurrenceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ValidateAttestationOccurrenceResponse) The response message.
      """
    config = self.GetMethodConfig('ValidateAttestationOccurrence')
    return self._RunMethod(config, request, global_params=global_params)