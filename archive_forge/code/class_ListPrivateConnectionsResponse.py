from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListPrivateConnectionsResponse(_messages.Message):
    """Response message for VmwareEngine.ListPrivateConnections

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    privateConnections: A list of private connections.
    unreachable: Unreachable resources.
  """
    nextPageToken = _messages.StringField(1)
    privateConnections = _messages.MessageField('PrivateConnection', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)