from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsMigrationJobsCreateRequest(_messages.Message):
    """A DatamigrationProjectsLocationsMigrationJobsCreateRequest object.

  Fields:
    migrationJob: A MigrationJob resource to be passed as the request body.
    migrationJobId: Required. The ID of the instance to create.
    parent: Required. The parent, which owns this collection of migration
      jobs.
    requestId: A unique id used to identify the request. If the server
      receives two requests with the same id, then the second request will be
      ignored. It is recommended to always set this value to a UUID. The id
      must contain only letters (a-z, A-Z), numbers (0-9), underscores (_),
      and hyphens (-). The maximum length is 40 characters.
  """
    migrationJob = _messages.MessageField('MigrationJob', 1)
    migrationJobId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)