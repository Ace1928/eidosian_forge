from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1alpha2 import healthcare_v1alpha2_messages as messages
def DeidentifyFhirResource(self, request, global_params=None):
    """De-identify a single FHIR resource.

      Args:
        request: (HealthcareProjectsLocationsServicesDeidentifyDeidentifyFhirResourceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpBody) The response message.
      """
    config = self.GetMethodConfig('DeidentifyFhirResource')
    return self._RunMethod(config, request, global_params=global_params)