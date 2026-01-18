from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsPrivateConnectionsCreateRequest(_messages.Message):
    """A DatamigrationProjectsLocationsPrivateConnectionsCreateRequest object.

  Fields:
    parent: Required. The parent that owns the collection of
      PrivateConnections.
    privateConnection: A PrivateConnection resource to be passed as the
      request body.
    privateConnectionId: Required. The private connection identifier.
    requestId: Optional. A unique ID used to identify the request. If the
      server receives two requests with the same ID, then the second request
      is ignored. It is recommended to always set this value to a UUID. The ID
      must contain only letters (a-z, A-Z), numbers (0-9), underscores (_),
      and hyphens (-). The maximum length is 40 characters.
    skipValidation: Optional. If set to true, will skip validations.
  """
    parent = _messages.StringField(1, required=True)
    privateConnection = _messages.MessageField('PrivateConnection', 2)
    privateConnectionId = _messages.StringField(3)
    requestId = _messages.StringField(4)
    skipValidation = _messages.BooleanField(5)