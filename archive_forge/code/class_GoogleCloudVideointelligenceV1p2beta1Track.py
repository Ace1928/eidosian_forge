from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1p2beta1Track(_messages.Message):
    """A track of an object instance.

  Fields:
    attributes: Optional. Attributes in the track level.
    confidence: Optional. The confidence score of the tracked object.
    segment: Video segment of a track.
    timestampedObjects: The object with timestamp and attributes per frame in
      the track.
  """
    attributes = _messages.MessageField('GoogleCloudVideointelligenceV1p2beta1DetectedAttribute', 1, repeated=True)
    confidence = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    segment = _messages.MessageField('GoogleCloudVideointelligenceV1p2beta1VideoSegment', 3)
    timestampedObjects = _messages.MessageField('GoogleCloudVideointelligenceV1p2beta1TimestampedObject', 4, repeated=True)