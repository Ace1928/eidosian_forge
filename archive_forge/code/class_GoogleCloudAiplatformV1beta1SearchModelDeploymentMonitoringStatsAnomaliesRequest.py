from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SearchModelDeploymentMonitoringStatsAnomaliesRequest(_messages.Message):
    """Request message for
  JobService.SearchModelDeploymentMonitoringStatsAnomalies.

  Fields:
    deployedModelId: Required. The DeployedModel ID of the
      [ModelDeploymentMonitoringObjectiveConfig.deployed_model_id].
    endTime: The latest timestamp of stats being generated. If not set,
      indicates feching stats till the latest possible one.
    featureDisplayName: The feature display name. If specified, only return
      the stats belonging to this feature. Format: ModelMonitoringStatsAnomali
      es.FeatureHistoricStatsAnomalies.feature_display_name, example:
      "user_destination".
    objectives: Required. Objectives of the stats to retrieve.
    pageSize: The standard list page size.
    pageToken: A page token received from a previous
      JobService.SearchModelDeploymentMonitoringStatsAnomalies call.
    startTime: The earliest timestamp of stats being generated. If not set,
      indicates fetching stats till the earliest possible one.
  """
    deployedModelId = _messages.StringField(1)
    endTime = _messages.StringField(2)
    featureDisplayName = _messages.StringField(3)
    objectives = _messages.MessageField('GoogleCloudAiplatformV1beta1SearchModelDeploymentMonitoringStatsAnomaliesRequestStatsAnomaliesObjective', 4, repeated=True)
    pageSize = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(6)
    startTime = _messages.StringField(7)