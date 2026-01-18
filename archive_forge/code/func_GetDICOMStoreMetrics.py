from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1alpha2 import healthcare_v1alpha2_messages as messages
def GetDICOMStoreMetrics(self, request, global_params=None):
    """Gets metrics associated with the DICOM store.

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresGetDICOMStoreMetricsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DicomStoreMetrics) The response message.
      """
    config = self.GetMethodConfig('GetDICOMStoreMetrics')
    return self._RunMethod(config, request, global_params=global_params)