from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1p1beta1LabelSegment(_messages.Message):
    """Video segment level annotation results for label detection.

  Fields:
    confidence: Confidence that the label is accurate. Range: [0, 1].
    segment: Video segment where a label was detected.
  """
    confidence = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    segment = _messages.MessageField('GoogleCloudVideointelligenceV1p1beta1VideoSegment', 2)