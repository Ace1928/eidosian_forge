from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsMigrationJobsRestartRequest(_messages.Message):
    """A DatamigrationProjectsLocationsMigrationJobsRestartRequest object.

  Fields:
    name: Name of the migration job resource to restart.
    restartMigrationJobRequest: A RestartMigrationJobRequest resource to be
      passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    restartMigrationJobRequest = _messages.MessageField('RestartMigrationJobRequest', 2)