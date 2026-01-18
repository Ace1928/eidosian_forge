from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SearchModelDeploymentMonitoringStatsAnomaliesResponse(_messages.Message):
    """Response message for
  JobService.SearchModelDeploymentMonitoringStatsAnomalies.

  Fields:
    monitoringStats: Stats retrieved for requested objectives. There are at
      most 1000 ModelMonitoringStatsAnomalies.FeatureHistoricStatsAnomalies.pr
      ediction_stats in the response.
    nextPageToken: The page token that can be used by the next
      JobService.SearchModelDeploymentMonitoringStatsAnomalies call.
  """
    monitoringStats = _messages.MessageField('GoogleCloudAiplatformV1beta1ModelMonitoringStatsAnomalies', 1, repeated=True)
    nextPageToken = _messages.StringField(2)