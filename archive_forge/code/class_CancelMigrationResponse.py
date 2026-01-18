from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CancelMigrationResponse(_messages.Message):
    """Response message for DataprocMetastore.CancelMigration.

  Fields:
    migrationExecution: The relative resource name of the migration execution,
      in the following form:projects/{project_number}/locations/{location_id}/
      services/{service_id}/migrationExecutions/{migration_execution_id}.
  """
    migrationExecution = _messages.StringField(1)