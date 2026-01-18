from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1 import healthcare_v1_messages as messages
def GetStudyMetrics(self, request, global_params=None):
    """GetStudyMetrics returns metrics for a study.

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresDicomWebStudiesGetStudyMetricsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StudyMetrics) The response message.
      """
    config = self.GetMethodConfig('GetStudyMetrics')
    return self._RunMethod(config, request, global_params=global_params)