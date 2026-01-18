from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsMigrationJobsDeleteRequest(_messages.Message):
    """A DatamigrationProjectsLocationsMigrationJobsDeleteRequest object.

  Fields:
    force: The destination CloudSQL connection profile is always deleted with
      the migration job. In case of force delete, the destination CloudSQL
      replica database is also deleted.
    name: Required. Name of the migration job resource to delete.
    requestId: A unique id used to identify the request. If the server
      receives two requests with the same id, then the second request will be
      ignored. It is recommended to always set this value to a UUID. The id
      must contain only letters (a-z, A-Z), numbers (0-9), underscores (_),
      and hyphens (-). The maximum length is 40 characters.
  """
    force = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)
    requestId = _messages.StringField(3)