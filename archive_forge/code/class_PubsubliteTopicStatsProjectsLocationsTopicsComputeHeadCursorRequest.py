from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PubsubliteTopicStatsProjectsLocationsTopicsComputeHeadCursorRequest(_messages.Message):
    """A PubsubliteTopicStatsProjectsLocationsTopicsComputeHeadCursorRequest
  object.

  Fields:
    computeHeadCursorRequest: A ComputeHeadCursorRequest resource to be passed
      as the request body.
    topic: Required. The topic for which we should compute the head cursor.
  """
    computeHeadCursorRequest = _messages.MessageField('ComputeHeadCursorRequest', 1)
    topic = _messages.StringField(2, required=True)