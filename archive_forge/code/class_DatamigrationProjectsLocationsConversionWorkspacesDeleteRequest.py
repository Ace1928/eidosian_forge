from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsConversionWorkspacesDeleteRequest(_messages.Message):
    """A DatamigrationProjectsLocationsConversionWorkspacesDeleteRequest
  object.

  Fields:
    force: Force delete the conversion workspace, even if there's a running
      migration that is using the workspace.
    name: Required. Name of the conversion workspace resource to delete.
    requestId: A unique ID used to identify the request. If the server
      receives two requests with the same ID, then the second request is
      ignored. It is recommended to always set this value to a UUID. The ID
      must contain only letters (a-z, A-Z), numbers (0-9), underscores (_),
      and hyphens (-). The maximum length is 40 characters.
  """
    force = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)
    requestId = _messages.StringField(3)