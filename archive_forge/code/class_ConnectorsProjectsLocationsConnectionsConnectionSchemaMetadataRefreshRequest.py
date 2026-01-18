from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectorsProjectsLocationsConnectionsConnectionSchemaMetadataRefreshRequest(_messages.Message):
    """A
  ConnectorsProjectsLocationsConnectionsConnectionSchemaMetadataRefreshRequest
  object.

  Fields:
    name: Required. Resource name. Format: projects/{project}/locations/{locat
      ion}/connections/{connection}/connectionSchemaMetadata
    refreshConnectionSchemaMetadataRequest: A
      RefreshConnectionSchemaMetadataRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    refreshConnectionSchemaMetadataRequest = _messages.MessageField('RefreshConnectionSchemaMetadataRequest', 2)