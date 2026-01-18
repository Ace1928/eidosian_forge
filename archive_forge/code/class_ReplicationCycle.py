from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReplicationCycle(_messages.Message):
    """ReplicationCycle contains information about the current replication
  cycle status.

  Enums:
    StateValueValuesEnum: State of the ReplicationCycle.

  Fields:
    cycleNumber: The cycle's ordinal number.
    endTime: The time the replication cycle has ended.
    error: Provides details on the state of the cycle in case of an error.
    name: The identifier of the ReplicationCycle.
    progressPercent: The current progress in percentage of this cycle. Was
      replaced by 'steps' field, which breaks down the cycle progression more
      accurately.
    startTime: The time the replication cycle has started.
    state: State of the ReplicationCycle.
    steps: The cycle's steps list representing its progress.
    totalPauseDuration: The accumulated duration the replication cycle was
      paused.
    warnings: Output only. Warnings that occurred during the cycle.
  """

    class StateValueValuesEnum(_messages.Enum):
        """State of the ReplicationCycle.

    Values:
      STATE_UNSPECIFIED: The state is unknown. This is used for API
        compatibility only and is not used by the system.
      RUNNING: The replication cycle is running.
      PAUSED: The replication cycle is paused.
      FAILED: The replication cycle finished with errors.
      SUCCEEDED: The replication cycle finished successfully.
    """
        STATE_UNSPECIFIED = 0
        RUNNING = 1
        PAUSED = 2
        FAILED = 3
        SUCCEEDED = 4
    cycleNumber = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    endTime = _messages.StringField(2)
    error = _messages.MessageField('Status', 3)
    name = _messages.StringField(4)
    progressPercent = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    startTime = _messages.StringField(6)
    state = _messages.EnumField('StateValueValuesEnum', 7)
    steps = _messages.MessageField('CycleStep', 8, repeated=True)
    totalPauseDuration = _messages.StringField(9)
    warnings = _messages.MessageField('MigrationWarning', 10, repeated=True)