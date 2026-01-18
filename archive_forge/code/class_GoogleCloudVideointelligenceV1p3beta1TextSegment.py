from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1p3beta1TextSegment(_messages.Message):
    """Video segment level annotation results for text detection.

  Fields:
    confidence: Confidence for the track of detected text. It is calculated as
      the highest over all frames where OCR detected text appears.
    frames: Information related to the frames where OCR detected text appears.
    segment: Video segment where a text snippet was detected.
  """
    confidence = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    frames = _messages.MessageField('GoogleCloudVideointelligenceV1p3beta1TextFrame', 2, repeated=True)
    segment = _messages.MessageField('GoogleCloudVideointelligenceV1p3beta1VideoSegment', 3)