from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1beta1 import healthcare_v1beta1_messages as messages
def Patient_consent_enforcement_status(self, request, global_params=None):
    """Returns the consent enforcement status of all consent resources for a patient. On success, the response body contains a JSON-encoded representation of a bundle of `Parameters` (http://hl7.org/fhir/parameters.html) FHIR resources, containing the current enforcement status for each consent resource of the patient. Does not support DSTU2.

      Args:
        request: (HealthcareProjectsLocationsDatasetsFhirStoresFhirPatientConsentEnforcementStatusRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpBody) The response message.
      """
    config = self.GetMethodConfig('Patient_consent_enforcement_status')
    return self._RunMethod(config, request, global_params=global_params)