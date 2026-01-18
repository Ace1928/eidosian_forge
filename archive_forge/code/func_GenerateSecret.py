from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sasportal.v1alpha1 import sasportal_v1alpha1_messages as messages
def GenerateSecret(self, request, global_params=None):
    """Generates a secret to be used with the ValidateInstaller.

      Args:
        request: (SasPortalGenerateSecretRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalGenerateSecretResponse) The response message.
      """
    config = self.GetMethodConfig('GenerateSecret')
    return self._RunMethod(config, request, global_params=global_params)