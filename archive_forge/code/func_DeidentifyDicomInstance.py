from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1alpha2 import healthcare_v1alpha2_messages as messages
def DeidentifyDicomInstance(self, request, global_params=None):
    """De-identify a single DICOM instance. Uses the ATTRIBUTE_CONFIDENTIALITY_BASIC_PROFILE TagFilterProfile and the REDACT_ALL_TEXT TextRedactionMode.

      Args:
        request: (HealthcareProjectsLocationsServicesDeidentifyDeidentifyDicomInstanceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpBody) The response message.
      """
    config = self.GetMethodConfig('DeidentifyDicomInstance')
    return self._RunMethod(config, request, global_params=global_params)