from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ComputeEnvironmentScoresResponse(_messages.Message):
    """Response for ComputeEnvironmentScores.

  Fields:
    nextPageToken: A page token, received from a previous `ComputeScore` call.
      Provide this to retrieve the subsequent page.
    scores: List of scores. One score per day.
  """
    nextPageToken = _messages.StringField(1)
    scores = _messages.MessageField('GoogleCloudApigeeV1Score', 2, repeated=True)