from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1p1beta1LabelFrame(_messages.Message):
    """Video frame level annotation results for label detection.

  Fields:
    confidence: Confidence that the label is accurate. Range: [0, 1].
    timeOffset: Time-offset, relative to the beginning of the video,
      corresponding to the video frame for this location.
  """
    confidence = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    timeOffset = _messages.StringField(2)