from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaPredictPredictionTextSentimentPredictionResult(_messages.Message):
    """Prediction output format for Text Sentiment

  Fields:
    sentiment: The integer sentiment labels between 0 (inclusive) and
      sentimentMax label (inclusive), while 0 maps to the least positive
      sentiment and sentimentMax maps to the most positive one. The higher the
      score is, the more positive the sentiment in the text snippet is. Note:
      sentimentMax is an integer value between 1 (inclusive) and 10
      (inclusive).
  """
    sentiment = _messages.IntegerField(1, variant=_messages.Variant.INT32)