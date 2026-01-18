from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1LogoRecognitionAnnotation(_messages.Message):
    """Annotation corresponding to one detected, tracked and recognized logo
  class.

  Fields:
    entity: Entity category information to specify the logo class that all the
      logo tracks within this LogoRecognitionAnnotation are recognized as.
    segments: All video segments where the recognized logo appears. There
      might be multiple instances of the same logo class appearing in one
      VideoSegment.
    tracks: All logo tracks where the recognized logo appears. Each track
      corresponds to one logo instance appearing in consecutive frames.
  """
    entity = _messages.MessageField('GoogleCloudVideointelligenceV1Entity', 1)
    segments = _messages.MessageField('GoogleCloudVideointelligenceV1VideoSegment', 2, repeated=True)
    tracks = _messages.MessageField('GoogleCloudVideointelligenceV1Track', 3, repeated=True)