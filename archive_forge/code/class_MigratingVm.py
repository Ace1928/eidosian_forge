from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MigratingVm(_messages.Message):
    """MigratingVm describes the VM that will be migrated from a Source
  environment and its replication state.

  Enums:
    StateValueValuesEnum: Output only. State of the MigratingVm.

  Messages:
    LabelsValue: The labels of the migrating VM.

  Fields:
    awsSourceVmDetails: Output only. Details of the VM from an AWS source.
    azureSourceVmDetails: Output only. Details of the VM from an Azure source.
    computeEngineDisksTargetDefaults: Details of the target Persistent Disks
      in Compute Engine.
    computeEngineTargetDefaults: Details of the target VM in Compute Engine.
    createTime: Output only. The time the migrating VM was created (this
      refers to this resource and not to the time it was installed in the
      source).
    currentSyncInfo: Output only. Details of the current running replication
      cycle.
    cutoverForecast: Output only. Provides details of future CutoverJobs of a
      MigratingVm. Set to empty when cutover forecast is unavailable.
    description: The description attached to the migrating VM by the user.
    displayName: The display name attached to the MigratingVm by the user.
    error: Output only. Provides details on the state of the Migrating VM in
      case of an error in replication.
    group: Output only. The group this migrating vm is included in, if any.
      The group is represented by the full path of the appropriate Group
      resource.
    labels: The labels of the migrating VM.
    lastReplicationCycle: Output only. Details of the last replication cycle.
      This will be updated whenever a replication cycle is finished and is not
      to be confused with last_sync which is only updated on successful
      replication cycles.
    lastSync: Output only. The most updated snapshot created time in the
      source that finished replication.
    name: Output only. The identifier of the MigratingVm.
    policy: The replication schedule policy.
    recentCloneJobs: Output only. The recent clone jobs performed on the
      migrating VM. This field holds the vm's last completed clone job and the
      vm's running clone job, if one exists. Note: To have this field
      populated you need to explicitly request it via the "view" parameter of
      the Get/List request.
    recentCutoverJobs: Output only. The recent cutover jobs performed on the
      migrating VM. This field holds the vm's last completed cutover job and
      the vm's running cutover job, if one exists. Note: To have this field
      populated you need to explicitly request it via the "view" parameter of
      the Get/List request.
    sourceVmId: The unique ID of the VM in the source. The VM's name in
      vSphere can be changed, so this is not the VM's name but rather its
      moRef id. This id is of the form vm-.
    state: Output only. State of the MigratingVm.
    stateTime: Output only. The last time the migrating VM state was updated.
    updateTime: Output only. The last time the migrating VM resource was
      updated.
    vmwareSourceVmDetails: Output only. Details of the VM from a Vmware
      source.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the MigratingVm.

    Values:
      STATE_UNSPECIFIED: The state was not sampled by the health checks yet.
      PENDING: The VM in the source is being verified.
      READY: The source VM was verified, and it's ready to start replication.
      FIRST_SYNC: Migration is going through the first sync cycle.
      ACTIVE: The replication is active, and it's running or scheduled to run.
      CUTTING_OVER: The source VM is being turned off, and a final replication
        is currently running.
      CUTOVER: The source VM was stopped and replicated. The replication is
        currently paused.
      FINAL_SYNC: A cutover job is active and replication cycle is running the
        final sync.
      PAUSED: The replication was paused by the user and no cycles are
        scheduled to run.
      FINALIZING: The migrating VM is being finalized and migration resources
        are being removed.
      FINALIZED: The replication process is done. The migrating VM is
        finalized and no longer consumes billable resources.
      ERROR: The replication process encountered an unrecoverable error and
        was aborted.
    """
        STATE_UNSPECIFIED = 0
        PENDING = 1
        READY = 2
        FIRST_SYNC = 3
        ACTIVE = 4
        CUTTING_OVER = 5
        CUTOVER = 6
        FINAL_SYNC = 7
        PAUSED = 8
        FINALIZING = 9
        FINALIZED = 10
        ERROR = 11

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The labels of the migrating VM.

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
    awsSourceVmDetails = _messages.MessageField('AwsSourceVmDetails', 1)
    azureSourceVmDetails = _messages.MessageField('AzureSourceVmDetails', 2)
    computeEngineDisksTargetDefaults = _messages.MessageField('ComputeEngineDisksTargetDefaults', 3)
    computeEngineTargetDefaults = _messages.MessageField('ComputeEngineTargetDefaults', 4)
    createTime = _messages.StringField(5)
    currentSyncInfo = _messages.MessageField('ReplicationCycle', 6)
    cutoverForecast = _messages.MessageField('CutoverForecast', 7)
    description = _messages.StringField(8)
    displayName = _messages.StringField(9)
    error = _messages.MessageField('Status', 10)
    group = _messages.StringField(11)
    labels = _messages.MessageField('LabelsValue', 12)
    lastReplicationCycle = _messages.MessageField('ReplicationCycle', 13)
    lastSync = _messages.MessageField('ReplicationSync', 14)
    name = _messages.StringField(15)
    policy = _messages.MessageField('SchedulePolicy', 16)
    recentCloneJobs = _messages.MessageField('CloneJob', 17, repeated=True)
    recentCutoverJobs = _messages.MessageField('CutoverJob', 18, repeated=True)
    sourceVmId = _messages.StringField(19)
    state = _messages.EnumField('StateValueValuesEnum', 20)
    stateTime = _messages.StringField(21)
    updateTime = _messages.StringField(22)
    vmwareSourceVmDetails = _messages.MessageField('VmwareSourceVmDetails', 23)