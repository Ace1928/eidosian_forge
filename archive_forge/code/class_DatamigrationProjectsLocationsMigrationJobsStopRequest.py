from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsMigrationJobsStopRequest(_messages.Message):
    """A DatamigrationProjectsLocationsMigrationJobsStopRequest object.

  Fields:
    name: Name of the migration job resource to stop.
    stopMigrationJobRequest: A StopMigrationJobRequest resource to be passed
      as the request body.
  """
    name = _messages.StringField(1, required=True)
    stopMigrationJobRequest = _messages.MessageField('StopMigrationJobRequest', 2)