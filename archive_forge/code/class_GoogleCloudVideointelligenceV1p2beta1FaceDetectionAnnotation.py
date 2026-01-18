from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1p2beta1FaceDetectionAnnotation(_messages.Message):
    """Face detection annotation.

  Fields:
    thumbnail: The thumbnail of a person's face.
    tracks: The face tracks with attributes.
    version: Feature version.
  """
    thumbnail = _messages.BytesField(1)
    tracks = _messages.MessageField('GoogleCloudVideointelligenceV1p2beta1Track', 2, repeated=True)
    version = _messages.StringField(3)