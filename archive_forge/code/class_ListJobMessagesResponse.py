from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListJobMessagesResponse(_messages.Message):
    """Response to a request to list job messages.

  Fields:
    autoscalingEvents: Autoscaling events in ascending timestamp order.
    jobMessages: Messages in ascending timestamp order.
    nextPageToken: The token to obtain the next page of results if there are
      more.
  """
    autoscalingEvents = _messages.MessageField('AutoscalingEvent', 1, repeated=True)
    jobMessages = _messages.MessageField('JobMessage', 2, repeated=True)
    nextPageToken = _messages.StringField(3)