from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1p3beta1StreamingVideoAnnotationResults(_messages.Message):
    """Streaming annotation results corresponding to a portion of the video
  that is currently being processed. Only ONE type of annotation will be
  specified in the response.

  Fields:
    explicitAnnotation: Explicit content annotation results.
    frameTimestamp: Timestamp of the processed frame in microseconds.
    labelAnnotations: Label annotation results.
    objectAnnotations: Object tracking results.
    shotAnnotations: Shot annotation results. Each shot is represented as a
      video segment.
  """
    explicitAnnotation = _messages.MessageField('GoogleCloudVideointelligenceV1p3beta1ExplicitContentAnnotation', 1)
    frameTimestamp = _messages.StringField(2)
    labelAnnotations = _messages.MessageField('GoogleCloudVideointelligenceV1p3beta1LabelAnnotation', 3, repeated=True)
    objectAnnotations = _messages.MessageField('GoogleCloudVideointelligenceV1p3beta1ObjectTrackingAnnotation', 4, repeated=True)
    shotAnnotations = _messages.MessageField('GoogleCloudVideointelligenceV1p3beta1VideoSegment', 5, repeated=True)