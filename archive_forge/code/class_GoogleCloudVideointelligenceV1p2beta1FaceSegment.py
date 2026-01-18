from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1p2beta1FaceSegment(_messages.Message):
    """Video segment level annotation results for face detection.

  Fields:
    segment: Video segment where a face was detected.
  """
    segment = _messages.MessageField('GoogleCloudVideointelligenceV1p2beta1VideoSegment', 1)