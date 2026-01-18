from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ModelMonitoringObjectiveConfig(_messages.Message):
    """The objective configuration for model monitoring, including the
  information needed to detect anomalies for one particular model.

  Fields:
    explanationConfig: The config for integrating with Vertex Explainable AI.
    predictionDriftDetectionConfig: The config for drift of prediction data.
    trainingDataset: Training dataset for models. This field has to be set
      only if TrainingPredictionSkewDetectionConfig is specified.
    trainingPredictionSkewDetectionConfig: The config for skew between
      training data and prediction data.
  """
    explanationConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1ModelMonitoringObjectiveConfigExplanationConfig', 1)
    predictionDriftDetectionConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1ModelMonitoringObjectiveConfigPredictionDriftDetectionConfig', 2)
    trainingDataset = _messages.MessageField('GoogleCloudAiplatformV1beta1ModelMonitoringObjectiveConfigTrainingDataset', 3)
    trainingPredictionSkewDetectionConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1ModelMonitoringObjectiveConfigTrainingPredictionSkewDetectionConfig', 4)