from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1beta1 import healthcare_v1beta1_messages as messages
def ApplyConsents(self, request, global_params=None):
    """Apply the Consent resources for the FHIR store and reindex the underlying resources in the FHIR store according to the aggregate consent. The aggregate consent of the patient in scope in this request replaces any previous call of this method. Any Consent resource change after this operation execution (including deletion) requires you to call ApplyConsents again to have effect. This method returns an Operation that can be used to track the progress of the consent resources that were processed by calling GetOperation. Upon completion, the ApplyConsentsResponse additionally contains the number of resources that was reindexed. Errors are logged to Cloud Logging (see [Viewing error logs in Cloud Logging](https://cloud.google.com/healthcare/docs/how-tos/logging)). To enforce consent check for data access, `consent_config.access_enforced` must be set to true for the FhirStore.

      Args:
        request: (HealthcareProjectsLocationsDatasetsFhirStoresApplyConsentsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('ApplyConsents')
    return self._RunMethod(config, request, global_params=global_params)