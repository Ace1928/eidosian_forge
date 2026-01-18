from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PubsubliteTopicStatsProjectsLocationsTopicsComputeTimeCursorRequest(_messages.Message):
    """A PubsubliteTopicStatsProjectsLocationsTopicsComputeTimeCursorRequest
  object.

  Fields:
    computeTimeCursorRequest: A ComputeTimeCursorRequest resource to be passed
      as the request body.
    topic: Required. The topic for which we should compute the cursor.
  """
    computeTimeCursorRequest = _messages.MessageField('ComputeTimeCursorRequest', 1)
    topic = _messages.StringField(2, required=True)