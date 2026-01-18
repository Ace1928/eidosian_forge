from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaPredictInstanceTextSentimentPredictionInstance(_messages.Message):
    """Prediction input format for Text Sentiment.

  Fields:
    content: The text snippet to make the predictions on.
    mimeType: The MIME type of the text snippet. The supported MIME types are
      listed below. - text/plain
  """
    content = _messages.StringField(1)
    mimeType = _messages.StringField(2)