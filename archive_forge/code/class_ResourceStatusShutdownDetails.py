from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceStatusShutdownDetails(_messages.Message):
    """Specifies if the instance is in `SHUTTING_DOWN` state or there is a
  instance stopping scheduled.

  Enums:
    StopStateValueValuesEnum: Current stopping state of the instance.
    TargetStateValueValuesEnum: Target instance state.

  Fields:
    maxDuration: Duration for graceful shutdown. Only applicable when
      `stop_state=SHUTTING_DOWN`.
    requestTimestamp: Past timestamp indicating the beginning of current
      `stopState` in RFC3339 text format.
    stopState: Current stopping state of the instance.
    targetState: Target instance state.
  """

    class StopStateValueValuesEnum(_messages.Enum):
        """Current stopping state of the instance.

    Values:
      SHUTTING_DOWN: The instance is gracefully shutting down.
      STOPPING: The instance is stopping.
    """
        SHUTTING_DOWN = 0
        STOPPING = 1

    class TargetStateValueValuesEnum(_messages.Enum):
        """Target instance state.

    Values:
      DELETED: The instance will be deleted.
      STOPPED: The instance will be stopped.
    """
        DELETED = 0
        STOPPED = 1
    maxDuration = _messages.MessageField('Duration', 1)
    requestTimestamp = _messages.StringField(2)
    stopState = _messages.EnumField('StopStateValueValuesEnum', 3)
    targetState = _messages.EnumField('TargetStateValueValuesEnum', 4)