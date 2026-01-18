from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaPredictParamsVideoClassificationPredictionParams(_messages.Message):
    """Prediction model parameters for Video Classification.

  Fields:
    confidenceThreshold: The Model only returns predictions with at least this
      confidence score. Default value is 0.0
    maxPredictions: The Model only returns up to that many top, by confidence
      score, predictions per instance. If this number is very high, the Model
      may return fewer predictions. Default value is 10,000.
    oneSecIntervalClassification: Set to true to request classification for a
      video at one-second intervals. Vertex AI returns labels and their
      confidence scores for each second of the entire time segment of the
      video that user specified in the input WARNING: Model evaluation is not
      done for this classification type, the quality of it depends on the
      training data, but there are no metrics provided to describe that
      quality. Default value is false
    segmentClassification: Set to true to request segment-level
      classification. Vertex AI returns labels and their confidence scores for
      the entire time segment of the video that user specified in the input
      instance. Default value is true
    shotClassification: Set to true to request shot-level classification.
      Vertex AI determines the boundaries for each camera shot in the entire
      time segment of the video that user specified in the input instance.
      Vertex AI then returns labels and their confidence scores for each
      detected shot, along with the start and end time of the shot. WARNING:
      Model evaluation is not done for this classification type, the quality
      of it depends on the training data, but there are no metrics provided to
      describe that quality. Default value is false
  """
    confidenceThreshold = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    maxPredictions = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    oneSecIntervalClassification = _messages.BooleanField(3)
    segmentClassification = _messages.BooleanField(4)
    shotClassification = _messages.BooleanField(5)