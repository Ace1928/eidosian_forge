from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListRolloutsResponse(_messages.Message):
    """Response message for listing rollouts.

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    rollouts: The rollouts from the specified parent resource.
  """
    nextPageToken = _messages.StringField(1)
    rollouts = _messages.MessageField('Rollout', 2, repeated=True)