from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.recaptchaenterprise.v1 import recaptchaenterprise_v1_messages as messages
def RetrieveLegacySecretKey(self, request, global_params=None):
    """Returns the secret key related to the specified public key. You must use the legacy secret key only in a 3rd party integration with legacy reCAPTCHA.

      Args:
        request: (RecaptchaenterpriseProjectsKeysRetrieveLegacySecretKeyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1RetrieveLegacySecretKeyResponse) The response message.
      """
    config = self.GetMethodConfig('RetrieveLegacySecretKey')
    return self._RunMethod(config, request, global_params=global_params)