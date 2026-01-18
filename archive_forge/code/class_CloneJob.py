from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloneJob(_messages.Message):
    """CloneJob describes the process of creating a clone of a MigratingVM to
  the requested target based on the latest successful uploaded snapshots.
  While the migration cycles of a MigratingVm take place, it is possible to
  verify the uploaded VM can be started in the cloud, by creating a clone. The
  clone can be created without any downtime, and it is created using the
  latest snapshots which are already in the cloud. The cloneJob is only
  responsible for its work, not its products, which means once it is finished,
  it will never touch the instance it created. It will only delete it in case
  of the CloneJob being cancelled or upon failure to clone.

  Enums:
    StateValueValuesEnum: Output only. State of the clone job.

  Fields:
    computeEngineDisksTargetDetails: Output only. Details of the target
      Persistent Disks in Compute Engine.
    computeEngineTargetDetails: Output only. Details of the target VM in
      Compute Engine.
    createTime: Output only. The time the clone job was created (as an API
      call, not when it was actually created in the target).
    endTime: Output only. The time the clone job was ended.
    error: Output only. Provides details for the errors that led to the Clone
      Job's state.
    name: Output only. The name of the clone.
    state: Output only. State of the clone job.
    stateTime: Output only. The time the state was last updated.
    steps: Output only. The clone steps list representing its progress.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the clone job.

    Values:
      STATE_UNSPECIFIED: The state is unknown. This is used for API
        compatibility only and is not used by the system.
      PENDING: The clone job has not yet started.
      ACTIVE: The clone job is active and running.
      FAILED: The clone job finished with errors.
      SUCCEEDED: The clone job finished successfully.
      CANCELLED: The clone job was cancelled.
      CANCELLING: The clone job is being cancelled.
      ADAPTING_OS: OS adaptation is running as part of the clone job to
        generate license.
    """
        STATE_UNSPECIFIED = 0
        PENDING = 1
        ACTIVE = 2
        FAILED = 3
        SUCCEEDED = 4
        CANCELLED = 5
        CANCELLING = 6
        ADAPTING_OS = 7
    computeEngineDisksTargetDetails = _messages.MessageField('ComputeEngineDisksTargetDetails', 1)
    computeEngineTargetDetails = _messages.MessageField('ComputeEngineTargetDetails', 2)
    createTime = _messages.StringField(3)
    endTime = _messages.StringField(4)
    error = _messages.MessageField('Status', 5)
    name = _messages.StringField(6)
    state = _messages.EnumField('StateValueValuesEnum', 7)
    stateTime = _messages.StringField(8)
    steps = _messages.MessageField('CloneStep', 9, repeated=True)