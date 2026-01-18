from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.recaptchaenterprise.v1 import recaptchaenterprise_v1_messages as messages
def Annotate(self, request, global_params=None):
    """Annotates a previously created Assessment to provide additional information on whether the event turned out to be authentic or fraudulent.

      Args:
        request: (RecaptchaenterpriseProjectsAssessmentsAnnotateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1AnnotateAssessmentResponse) The response message.
      """
    config = self.GetMethodConfig('Annotate')
    return self._RunMethod(config, request, global_params=global_params)