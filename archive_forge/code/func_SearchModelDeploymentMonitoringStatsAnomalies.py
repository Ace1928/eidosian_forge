from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def SearchModelDeploymentMonitoringStatsAnomalies(self, request, global_params=None):
    """Searches Model Monitoring Statistics generated within a given time window.

      Args:
        request: (AiplatformProjectsLocationsModelDeploymentMonitoringJobsSearchModelDeploymentMonitoringStatsAnomaliesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1SearchModelDeploymentMonitoringStatsAnomaliesResponse) The response message.
      """
    config = self.GetMethodConfig('SearchModelDeploymentMonitoringStatsAnomalies')
    return self._RunMethod(config, request, global_params=global_params)