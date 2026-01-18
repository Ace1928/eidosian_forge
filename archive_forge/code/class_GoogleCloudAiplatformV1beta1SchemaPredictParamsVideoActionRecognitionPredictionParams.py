from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaPredictParamsVideoActionRecognitionPredictionParams(_messages.Message):
    """Prediction model parameters for Video Action Recognition.

  Fields:
    confidenceThreshold: The Model only returns predictions with at least this
      confidence score. Default value is 0.0
    maxPredictions: The model only returns up to that many top, by confidence
      score, predictions per frame of the video. If this number is very high,
      the Model may return fewer predictions per frame. Default value is 50.
  """
    confidenceThreshold = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    maxPredictions = _messages.IntegerField(2, variant=_messages.Variant.INT32)