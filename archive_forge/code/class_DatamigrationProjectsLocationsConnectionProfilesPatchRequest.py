from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsConnectionProfilesPatchRequest(_messages.Message):
    """A DatamigrationProjectsLocationsConnectionProfilesPatchRequest object.

  Fields:
    connectionProfile: A ConnectionProfile resource to be passed as the
      request body.
    name: The name of this connection profile resource in the form of projects
      /{project}/locations/{location}/connectionProfiles/{connectionProfile}.
    requestId: A unique id used to identify the request. If the server
      receives two requests with the same id, then the second request will be
      ignored. It is recommended to always set this value to a UUID. The id
      must contain only letters (a-z, A-Z), numbers (0-9), underscores (_),
      and hyphens (-). The maximum length is 40 characters.
    updateMask: Required. Field mask is used to specify the fields to be
      overwritten in the connection profile resource by the update.
  """
    connectionProfile = _messages.MessageField('ConnectionProfile', 1)
    name = _messages.StringField(2, required=True)
    requestId = _messages.StringField(3)
    updateMask = _messages.StringField(4)