from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaTimeSegment(_messages.Message):
    """A time period inside of a DataItem that has a time dimension (e.g.
  video).

  Fields:
    endTimeOffset: End of the time segment (exclusive), represented as the
      duration since the start of the DataItem.
    startTimeOffset: Start of the time segment (inclusive), represented as the
      duration since the start of the DataItem.
  """
    endTimeOffset = _messages.StringField(1)
    startTimeOffset = _messages.StringField(2)