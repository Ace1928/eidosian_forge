from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MigrationExecution(_messages.Message):
    """The details of a migration execution resource.

  Enums:
    PhaseValueValuesEnum: Output only. The current phase of the migration
      execution.
    StateValueValuesEnum: Output only. The current state of the migration
      execution.

  Fields:
    cloudSqlMigrationConfig: Configuration information specific to migrating
      from self-managed hive metastore on Google Cloud using Cloud SQL as the
      backend database to Dataproc Metastore.
    createTime: Output only. The time when the migration execution was
      started.
    endTime: Output only. The time when the migration execution finished.
    name: Output only. The relative resource name of the migration execution,
      in the following form: projects/{project_number}/locations/{location_id}
      /services/{service_id}/migrationExecutions/{migration_execution_id}
    phase: Output only. The current phase of the migration execution.
    state: Output only. The current state of the migration execution.
    stateMessage: Output only. Additional information about the current state
      of the migration execution.
  """

    class PhaseValueValuesEnum(_messages.Enum):
        """Output only. The current phase of the migration execution.

    Values:
      PHASE_UNSPECIFIED: The phase of the migration execution is unknown.
      REPLICATION: Replication phase refers to the migration phase when
        Dataproc Metastore is running a pipeline to replicate changes in the
        customer database to its backend database. During this phase, Dataproc
        Metastore uses the customer database as the hive metastore backend
        database.
      CUTOVER: Cutover phase refers to the migration phase when Dataproc
        Metastore switches to using its own backend database. Migration enters
        this phase when customer is done migrating all their
        clusters/workloads to Dataproc Metastore and triggers
        CompleteMigration.
    """
        PHASE_UNSPECIFIED = 0
        REPLICATION = 1
        CUTOVER = 2

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of the migration execution.

    Values:
      STATE_UNSPECIFIED: The state of the migration execution is unknown.
      STARTING: The migration execution is starting.
      RUNNING: The migration execution is running.
      CANCELLING: The migration execution is in the process of being
        cancelled.
      AWAITING_USER_ACTION: The migration execution is awaiting user action.
      SUCCEEDED: The migration execution has completed successfully.
      FAILED: The migration execution has failed.
      CANCELLED: The migration execution is cancelled.
      DELETING: The migration execution is being deleted.
    """
        STATE_UNSPECIFIED = 0
        STARTING = 1
        RUNNING = 2
        CANCELLING = 3
        AWAITING_USER_ACTION = 4
        SUCCEEDED = 5
        FAILED = 6
        CANCELLED = 7
        DELETING = 8
    cloudSqlMigrationConfig = _messages.MessageField('CloudSQLMigrationConfig', 1)
    createTime = _messages.StringField(2)
    endTime = _messages.StringField(3)
    name = _messages.StringField(4)
    phase = _messages.EnumField('PhaseValueValuesEnum', 5)
    state = _messages.EnumField('StateValueValuesEnum', 6)
    stateMessage = _messages.StringField(7)