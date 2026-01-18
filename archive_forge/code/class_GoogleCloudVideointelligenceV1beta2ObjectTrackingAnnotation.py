from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1beta2ObjectTrackingAnnotation(_messages.Message):
    """Annotations corresponding to one tracked object.

  Fields:
    confidence: Object category's labeling confidence of this track.
    entity: Entity to specify the object category that this track is labeled
      as.
    frames: Information corresponding to all frames where this object track
      appears. Non-streaming batch mode: it may be one or multiple
      ObjectTrackingFrame messages in frames. Streaming mode: it can only be
      one ObjectTrackingFrame message in frames.
    segment: Non-streaming batch mode ONLY. Each object track corresponds to
      one video segment where it appears.
    trackId: Streaming mode ONLY. In streaming mode, we do not know the end
      time of a tracked object before it is completed. Hence, there is no
      VideoSegment info returned. Instead, we provide a unique identifiable
      integer track_id so that the customers can correlate the results of the
      ongoing ObjectTrackAnnotation of the same track_id over time.
    version: Feature version.
  """
    confidence = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    entity = _messages.MessageField('GoogleCloudVideointelligenceV1beta2Entity', 2)
    frames = _messages.MessageField('GoogleCloudVideointelligenceV1beta2ObjectTrackingFrame', 3, repeated=True)
    segment = _messages.MessageField('GoogleCloudVideointelligenceV1beta2VideoSegment', 4)
    trackId = _messages.IntegerField(5)
    version = _messages.StringField(6)