from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaPredictPredictionTabularClassificationPredictionResult(_messages.Message):
    """Prediction output format for Tabular Classification.

  Fields:
    classes: The name of the classes being classified, contains all possible
      values of the target column.
    scores: The model's confidence in each class being correct, higher value
      means higher confidence. The N-th score corresponds to the N-th class in
      classes.
  """
    classes = _messages.StringField(1, repeated=True)
    scores = _messages.FloatField(2, repeated=True, variant=_messages.Variant.FLOAT)