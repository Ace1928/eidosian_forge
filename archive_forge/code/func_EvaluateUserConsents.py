from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1alpha2 import healthcare_v1alpha2_messages as messages
def EvaluateUserConsents(self, request, global_params=None):
    """Evaluates the end user's Consents for all matching User data mappings. Note: User data mappings are indexed asynchronously, which can cause a slight delay between the time mappings are created or updated and when they are included in EvaluateUserConsents results.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresEvaluateUserConsentsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (EvaluateUserConsentsResponse) The response message.
      """
    config = self.GetMethodConfig('EvaluateUserConsents')
    return self._RunMethod(config, request, global_params=global_params)