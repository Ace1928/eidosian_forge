from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1FaceAnnotation(_messages.Message):
    """Deprecated. No effect.

  Fields:
    frames: All video frames where a face was detected.
    segments: All video segments where a face was detected.
    thumbnail: Thumbnail of a representative face view (in JPEG format).
  """
    frames = _messages.MessageField('GoogleCloudVideointelligenceV1FaceFrame', 1, repeated=True)
    segments = _messages.MessageField('GoogleCloudVideointelligenceV1FaceSegment', 2, repeated=True)
    thumbnail = _messages.BytesField(3)