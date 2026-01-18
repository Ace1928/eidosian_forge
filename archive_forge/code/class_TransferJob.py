from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransferJob(_messages.Message):
    """This resource represents the configuration of a transfer job that runs
  periodically.

  Enums:
    StatusValueValuesEnum: Status of the job. This value MUST be specified for
      `CreateTransferJobRequests`. **Note:** The effect of the new job status
      takes place during a subsequent job run. For example, if you change the
      job status from ENABLED to DISABLED, and an operation spawned by the
      transfer is running, the status change would not affect the current
      operation.

  Fields:
    creationTime: Output only. The time that the transfer job was created.
    deletionTime: Output only. The time that the transfer job was deleted.
    description: A description provided by the user for the job. Its max
      length is 1024 bytes when Unicode-encoded.
    eventStream: Specifies the event stream for the transfer job for event-
      driven transfers. When EventStream is specified, the Schedule fields are
      ignored.
    lastModificationTime: Output only. The time that the transfer job was last
      modified.
    latestOperationName: The name of the most recently started
      TransferOperation of this JobConfig. Present if a TransferOperation has
      been created for this JobConfig.
    loggingConfig: Logging configuration.
    name: A unique name (within the transfer project) assigned when the job is
      created. If this field is empty in a CreateTransferJobRequest, Storage
      Transfer Service assigns a unique name. Otherwise, the specified name is
      used as the unique name for this job. If the specified name is in use by
      a job, the creation request fails with an ALREADY_EXISTS error. This
      name must start with `"transferJobs/"` prefix and end with a letter or a
      number, and should be no more than 128 characters. For transfers
      involving PosixFilesystem, this name must start with `transferJobs/OPI`
      specifically. For all other transfer types, this name must not start
      with `transferJobs/OPI`. Non-PosixFilesystem example:
      `"transferJobs/^(?!OPI)[A-Za-z0-9-._~]*[A-Za-z0-9]$"` PosixFilesystem
      example: `"transferJobs/OPI^[A-Za-z0-9-._~]*[A-Za-z0-9]$"` Applications
      must not rely on the enforcement of naming requirements involving OPI.
      Invalid job names fail with an INVALID_ARGUMENT error.
    notificationConfig: Notification configuration.
    projectId: The ID of the Google Cloud project that owns the job.
    replicationSpec: Replication specification.
    schedule: Specifies schedule for the transfer job. This is an optional
      field. When the field is not set, the job never executes a transfer,
      unless you invoke RunTransferJob or update the job to have a non-empty
      schedule.
    status: Status of the job. This value MUST be specified for
      `CreateTransferJobRequests`. **Note:** The effect of the new job status
      takes place during a subsequent job run. For example, if you change the
      job status from ENABLED to DISABLED, and an operation spawned by the
      transfer is running, the status change would not affect the current
      operation.
    transferSpec: Transfer specification.
  """

    class StatusValueValuesEnum(_messages.Enum):
        """Status of the job. This value MUST be specified for
    `CreateTransferJobRequests`. **Note:** The effect of the new job status
    takes place during a subsequent job run. For example, if you change the
    job status from ENABLED to DISABLED, and an operation spawned by the
    transfer is running, the status change would not affect the current
    operation.

    Values:
      STATUS_UNSPECIFIED: Zero is an illegal value.
      ENABLED: New transfers are performed based on the schedule.
      DISABLED: New transfers are not scheduled.
      DELETED: This is a soft delete state. After a transfer job is set to
        this state, the job and all the transfer executions are subject to
        garbage collection. Transfer jobs become eligible for garbage
        collection 30 days after their status is set to `DELETED`.
    """
        STATUS_UNSPECIFIED = 0
        ENABLED = 1
        DISABLED = 2
        DELETED = 3
    creationTime = _messages.StringField(1)
    deletionTime = _messages.StringField(2)
    description = _messages.StringField(3)
    eventStream = _messages.MessageField('EventStream', 4)
    lastModificationTime = _messages.StringField(5)
    latestOperationName = _messages.StringField(6)
    loggingConfig = _messages.MessageField('LoggingConfig', 7)
    name = _messages.StringField(8)
    notificationConfig = _messages.MessageField('NotificationConfig', 9)
    projectId = _messages.StringField(10)
    replicationSpec = _messages.MessageField('ReplicationSpec', 11)
    schedule = _messages.MessageField('Schedule', 12)
    status = _messages.EnumField('StatusValueValuesEnum', 13)
    transferSpec = _messages.MessageField('TransferSpec', 14)