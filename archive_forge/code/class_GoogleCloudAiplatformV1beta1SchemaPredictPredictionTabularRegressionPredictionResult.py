from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaPredictPredictionTabularRegressionPredictionResult(_messages.Message):
    """Prediction output format for Tabular Regression.

  Fields:
    lowerBound: The lower bound of the prediction interval.
    quantilePredictions: Quantile predictions, in 1-1 correspondence with
      quantile_values.
    quantileValues: Quantile values.
    upperBound: The upper bound of the prediction interval.
    value: The regression value.
  """
    lowerBound = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    quantilePredictions = _messages.FloatField(2, repeated=True, variant=_messages.Variant.FLOAT)
    quantileValues = _messages.FloatField(3, repeated=True, variant=_messages.Variant.FLOAT)
    upperBound = _messages.FloatField(4, variant=_messages.Variant.FLOAT)
    value = _messages.FloatField(5, variant=_messages.Variant.FLOAT)