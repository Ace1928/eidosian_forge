from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MigrationJob(_messages.Message):
    """Represents a Database Migration Service migration job object.

  Enums:
    PhaseValueValuesEnum: Output only. The current migration job phase.
    StateValueValuesEnum: The current migration job state.
    TypeValueValuesEnum: Required. The migration job type.

  Messages:
    LabelsValue: The resource labels for migration job to use to annotate any
      related underlying resources such as Compute Engine VMs. An object
      containing a list of "key": "value" pairs. Example: `{ "name": "wrench",
      "mass": "1.3kg", "count": "3" }`.

  Fields:
    createTime: Output only. The timestamp when the migration job resource was
      created. A timestamp in RFC3339 UTC "Zulu" format, accurate to
      nanoseconds. Example: "2014-10-02T15:01:23.045123456Z".
    destination: Required. The resource name (URI) of the destination
      connection profile.
    destinationDatabase: The database engine type and provider of the
      destination.
    displayName: The migration job display name.
    dumpPath: The path to the dump file in Google Cloud Storage, in the
      format: (gs://[BUCKET_NAME]/[OBJECT_NAME]).
    duration: Output only. The duration of the migration job (in seconds). A
      duration in seconds with up to nine fractional digits, terminated by
      's'. Example: "3.5s".
    endTime: Output only. If the migration job is completed, the time when it
      was completed.
    error: Output only. The error details in case of state FAILED.
    labels: The resource labels for migration job to use to annotate any
      related underlying resources such as Compute Engine VMs. An object
      containing a list of "key": "value" pairs. Example: `{ "name": "wrench",
      "mass": "1.3kg", "count": "3" }`.
    name: The name (URI) of this migration job resource, in the form of:
      projects/{project}/locations/{location}/migrationJobs/{migrationJob}.
    phase: Output only. The current migration job phase.
    reverseSshConnectivity: The details needed to communicate to the source
      over Reverse SSH tunnel connectivity.
    source: Required. The resource name (URI) of the source connection
      profile.
    sourceDatabase: The database engine type and provider of the source.
    state: The current migration job state.
    staticIpConnectivity: static ip connectivity data (default, no additional
      details needed).
    type: Required. The migration job type.
    updateTime: Output only. The timestamp when the migration job resource was
      last updated. A timestamp in RFC3339 UTC "Zulu" format, accurate to
      nanoseconds. Example: "2014-10-02T15:01:23.045123456Z".
    vpcPeeringConnectivity: The details of the VPC network that the source
      database is located in.
  """

    class PhaseValueValuesEnum(_messages.Enum):
        """Output only. The current migration job phase.

    Values:
      PHASE_UNSPECIFIED: The phase of the migration job is unknown.
      FULL_DUMP: The migration job is in the full dump phase.
      CDC: The migration job is CDC phase.
      PROMOTE_IN_PROGRESS: The migration job is running the promote phase.
      WAITING_FOR_SOURCE_WRITES_TO_STOP: Only RDS flow - waiting for source
        writes to stop
      PREPARING_THE_DUMP: Only RDS flow - the sources writes stopped, waiting
        for dump to begin
    """
        PHASE_UNSPECIFIED = 0
        FULL_DUMP = 1
        CDC = 2
        PROMOTE_IN_PROGRESS = 3
        WAITING_FOR_SOURCE_WRITES_TO_STOP = 4
        PREPARING_THE_DUMP = 5

    class StateValueValuesEnum(_messages.Enum):
        """The current migration job state.

    Values:
      STATE_UNSPECIFIED: The state of the migration job is unknown.
      MAINTENANCE: The migration job is down for maintenance.
      DRAFT: The migration job is in draft mode and no resources are created.
      CREATING: The migration job is being created.
      NOT_STARTED: The migration job is created and not started.
      RUNNING: The migration job is running.
      FAILED: The migration job failed.
      COMPLETED: The migration job has been completed.
      DELETING: The migration job is being deleted.
      STOPPING: The migration job is being stopped.
      STOPPED: The migration job is currently stopped.
      DELETED: The migration job has been deleted.
      UPDATING: The migration job is being updated.
      STARTING: The migration job is starting.
      RESTARTING: The migration job is restarting.
      RESUMING: The migration job is resuming.
    """
        STATE_UNSPECIFIED = 0
        MAINTENANCE = 1
        DRAFT = 2
        CREATING = 3
        NOT_STARTED = 4
        RUNNING = 5
        FAILED = 6
        COMPLETED = 7
        DELETING = 8
        STOPPING = 9
        STOPPED = 10
        DELETED = 11
        UPDATING = 12
        STARTING = 13
        RESTARTING = 14
        RESUMING = 15

    class TypeValueValuesEnum(_messages.Enum):
        """Required. The migration job type.

    Values:
      TYPE_UNSPECIFIED: The type of the migration job is unknown.
      ONE_TIME: The migration job is a one time migration.
      CONTINUOUS: The migration job is a continuous migration.
    """
        TYPE_UNSPECIFIED = 0
        ONE_TIME = 1
        CONTINUOUS = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The resource labels for migration job to use to annotate any related
    underlying resources such as Compute Engine VMs. An object containing a
    list of "key": "value" pairs. Example: `{ "name": "wrench", "mass":
    "1.3kg", "count": "3" }`.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    createTime = _messages.StringField(1)
    destination = _messages.StringField(2)
    destinationDatabase = _messages.MessageField('DatabaseType', 3)
    displayName = _messages.StringField(4)
    dumpPath = _messages.StringField(5)
    duration = _messages.StringField(6)
    endTime = _messages.StringField(7)
    error = _messages.MessageField('Status', 8)
    labels = _messages.MessageField('LabelsValue', 9)
    name = _messages.StringField(10)
    phase = _messages.EnumField('PhaseValueValuesEnum', 11)
    reverseSshConnectivity = _messages.MessageField('ReverseSshConnectivity', 12)
    source = _messages.StringField(13)
    sourceDatabase = _messages.MessageField('DatabaseType', 14)
    state = _messages.EnumField('StateValueValuesEnum', 15)
    staticIpConnectivity = _messages.MessageField('StaticIpConnectivity', 16)
    type = _messages.EnumField('TypeValueValuesEnum', 17)
    updateTime = _messages.StringField(18)
    vpcPeeringConnectivity = _messages.MessageField('VpcPeeringConnectivity', 19)