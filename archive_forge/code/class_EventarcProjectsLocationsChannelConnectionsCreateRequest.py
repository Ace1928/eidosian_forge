from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventarcProjectsLocationsChannelConnectionsCreateRequest(_messages.Message):
    """A EventarcProjectsLocationsChannelConnectionsCreateRequest object.

  Fields:
    channelConnection: A ChannelConnection resource to be passed as the
      request body.
    channelConnectionId: Required. The user-provided ID to be assigned to the
      channel connection.
    parent: Required. The parent collection in which to add this channel
      connection.
  """
    channelConnection = _messages.MessageField('ChannelConnection', 1)
    channelConnectionId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)