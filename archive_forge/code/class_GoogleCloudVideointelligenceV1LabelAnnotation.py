from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1LabelAnnotation(_messages.Message):
    """Label annotation.

  Fields:
    categoryEntities: Common categories for the detected entity. For example,
      when the label is `Terrier`, the category is likely `dog`. And in some
      cases there might be more than one categories e.g., `Terrier` could also
      be a `pet`.
    entity: Detected entity.
    frames: All video frames where a label was detected.
    segments: All video segments where a label was detected.
    version: Feature version.
  """
    categoryEntities = _messages.MessageField('GoogleCloudVideointelligenceV1Entity', 1, repeated=True)
    entity = _messages.MessageField('GoogleCloudVideointelligenceV1Entity', 2)
    frames = _messages.MessageField('GoogleCloudVideointelligenceV1LabelFrame', 3, repeated=True)
    segments = _messages.MessageField('GoogleCloudVideointelligenceV1LabelSegment', 4, repeated=True)
    version = _messages.StringField(5)