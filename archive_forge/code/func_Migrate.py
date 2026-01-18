from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.recaptchaenterprise.v1 import recaptchaenterprise_v1_messages as messages
def Migrate(self, request, global_params=None):
    """Migrates an existing key from reCAPTCHA to reCAPTCHA Enterprise. Once a key is migrated, it can be used from either product. SiteVerify requests are billed as CreateAssessment calls. You must be authenticated as one of the current owners of the reCAPTCHA Key, and your user must have the reCAPTCHA Enterprise Admin IAM role in the destination project.

      Args:
        request: (RecaptchaenterpriseProjectsKeysMigrateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1Key) The response message.
      """
    config = self.GetMethodConfig('Migrate')
    return self._RunMethod(config, request, global_params=global_params)