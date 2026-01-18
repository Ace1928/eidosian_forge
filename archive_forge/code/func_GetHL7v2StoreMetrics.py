from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1alpha2 import healthcare_v1alpha2_messages as messages
def GetHL7v2StoreMetrics(self, request, global_params=None):
    """Gets metrics associated with the HL7v2 store.

      Args:
        request: (HealthcareProjectsLocationsDatasetsHl7V2StoresGetHL7v2StoreMetricsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Hl7V2StoreMetrics) The response message.
      """
    config = self.GetMethodConfig('GetHL7v2StoreMetrics')
    return self._RunMethod(config, request, global_params=global_params)