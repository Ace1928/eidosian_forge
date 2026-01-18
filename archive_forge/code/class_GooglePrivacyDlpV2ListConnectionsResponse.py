from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2ListConnectionsResponse(_messages.Message):
    """Response message for ListConnections.

  Fields:
    connections: List of connections.
    nextPageToken: Token to retrieve the next page of results. An empty value
      means there are no more results.
  """
    connections = _messages.MessageField('GooglePrivacyDlpV2Connection', 1, repeated=True)
    nextPageToken = _messages.StringField(2)