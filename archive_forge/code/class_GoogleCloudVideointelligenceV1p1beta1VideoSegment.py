from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1p1beta1VideoSegment(_messages.Message):
    """Video segment.

  Fields:
    endTimeOffset: Time-offset, relative to the beginning of the video,
      corresponding to the end of the segment (inclusive).
    startTimeOffset: Time-offset, relative to the beginning of the video,
      corresponding to the start of the segment (inclusive).
  """
    endTimeOffset = _messages.StringField(1)
    startTimeOffset = _messages.StringField(2)