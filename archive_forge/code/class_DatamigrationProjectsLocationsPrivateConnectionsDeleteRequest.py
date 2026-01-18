from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsPrivateConnectionsDeleteRequest(_messages.Message):
    """A DatamigrationProjectsLocationsPrivateConnectionsDeleteRequest object.

  Fields:
    name: Required. The name of the private connection to delete.
    requestId: Optional. A unique ID used to identify the request. If the
      server receives two requests with the same ID, then the second request
      is ignored. It is recommended to always set this value to a UUID. The ID
      must contain only letters (a-z, A-Z), numbers (0-9), underscores (_),
      and hyphens (-). The maximum length is 40 characters.
  """
    name = _messages.StringField(1, required=True)
    requestId = _messages.StringField(2)