from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsConnectionProfilesDeleteRequest(_messages.Message):
    """A DatamigrationProjectsLocationsConnectionProfilesDeleteRequest object.

  Fields:
    force: In case of force delete, the CloudSQL replica database is also
      deleted (only for CloudSQL connection profile).
    name: Required. Name of the connection profile resource to delete.
    requestId: A unique id used to identify the request. If the server
      receives two requests with the same id, then the second request will be
      ignored. It is recommended to always set this value to a UUID. The id
      must contain only letters (a-z, A-Z), numbers (0-9), underscores (_),
      and hyphens (-). The maximum length is 40 characters.
  """
    force = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)
    requestId = _messages.StringField(3)