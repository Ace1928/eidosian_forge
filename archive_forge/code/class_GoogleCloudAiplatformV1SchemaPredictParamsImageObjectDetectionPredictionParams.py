from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaPredictParamsImageObjectDetectionPredictionParams(_messages.Message):
    """Prediction model parameters for Image Object Detection.

  Fields:
    confidenceThreshold: The Model only returns predictions with at least this
      confidence score. Default value is 0.0
    maxPredictions: The Model only returns up to that many top, by confidence
      score, predictions per instance. Note that number of returned
      predictions is also limited by metadata's predictionsLimit. Default
      value is 10.
  """
    confidenceThreshold = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    maxPredictions = _messages.IntegerField(2, variant=_messages.Variant.INT32)