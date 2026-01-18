from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListServiceRolloutsResponse(_messages.Message):
    """Response message for ListServiceRollouts method.

  Fields:
    nextPageToken: The token of the next page of results.
    rollouts: The list of rollout resources.
  """
    nextPageToken = _messages.StringField(1)
    rollouts = _messages.MessageField('Rollout', 2, repeated=True)