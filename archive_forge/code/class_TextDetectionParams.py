from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TextDetectionParams(_messages.Message):
    """Parameters for text detections. This is used to control TEXT_DETECTION
  and DOCUMENT_TEXT_DETECTION features.

  Fields:
    advancedOcrOptions: A list of advanced OCR options to further fine-tune
      OCR behavior. Current valid values are: - `legacy_layout`: a heuristics
      layout detection algorithm, which serves as an alternative to the
      current ML-based layout detection algorithm. Customers can choose the
      best suitable layout algorithm based on their situation.
    enableTextDetectionConfidenceScore: By default, Cloud Vision API only
      includes confidence score for DOCUMENT_TEXT_DETECTION result. Set the
      flag to true to include confidence score for TEXT_DETECTION as well.
  """
    advancedOcrOptions = _messages.StringField(1, repeated=True)
    enableTextDetectionConfidenceScore = _messages.BooleanField(2)