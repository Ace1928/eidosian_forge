from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1alpha2 import healthcare_v1alpha2_messages as messages
def GetFHIRStoreMetrics(self, request, global_params=None):
    """Gets metrics associated with the FHIR store.

      Args:
        request: (HealthcareProjectsLocationsDatasetsFhirStoresGetFHIRStoreMetricsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FhirStoreMetrics) The response message.
      """
    config = self.GetMethodConfig('GetFHIRStoreMetrics')
    return self._RunMethod(config, request, global_params=global_params)