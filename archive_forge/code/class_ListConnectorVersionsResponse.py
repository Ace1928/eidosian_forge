from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListConnectorVersionsResponse(_messages.Message):
    """Response message for Connectors.ListConnectorVersions.

  Fields:
    connectorVersions: A list of connector versions.
    nextPageToken: Next page token.
    unreachable: Locations that could not be reached.
  """
    connectorVersions = _messages.MessageField('ConnectorVersion', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)