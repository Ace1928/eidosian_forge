from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListChannelsResponse(_messages.Message):
    """The response message for the `ListChannels` method.

  Fields:
    channels: The requested channels, up to the number specified in
      `page_size`.
    nextPageToken: A page token that can be sent to `ListChannels` to request
      the next page. If this is empty, then there are no more pages.
    unreachable: Unreachable resources, if any.
  """
    channels = _messages.MessageField('Channel', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)