from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ModelDeploymentMonitoringObjectiveConfig(_messages.Message):
    """ModelDeploymentMonitoringObjectiveConfig contains the pair of
  deployed_model_id to ModelMonitoringObjectiveConfig.

  Fields:
    deployedModelId: The DeployedModel ID of the objective config.
    objectiveConfig: The objective config of for the modelmonitoring job of
      this deployed model.
  """
    deployedModelId = _messages.StringField(1)
    objectiveConfig = _messages.MessageField('GoogleCloudAiplatformV1ModelMonitoringObjectiveConfig', 2)