from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsConnectionProfilesCreateRequest(_messages.Message):
    """A DatamigrationProjectsLocationsConnectionProfilesCreateRequest object.

  Fields:
    connectionProfile: A ConnectionProfile resource to be passed as the
      request body.
    connectionProfileId: Required. The connection profile identifier.
    parent: Required. The parent, which owns this collection of connection
      profiles.
    requestId: A unique id used to identify the request. If the server
      receives two requests with the same id, then the second request will be
      ignored. It is recommended to always set this value to a UUID. The id
      must contain only letters (a-z, A-Z), numbers (0-9), underscores (_),
      and hyphens (-). The maximum length is 40 characters.
  """
    connectionProfile = _messages.MessageField('ConnectionProfile', 1)
    connectionProfileId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)