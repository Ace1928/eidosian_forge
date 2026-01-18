from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutoscalingEvent(_messages.Message):
    """A structured message reporting an autoscaling decision made by the
  Dataflow service.

  Enums:
    EventTypeValueValuesEnum: The type of autoscaling event to report.

  Fields:
    currentNumWorkers: The current number of workers the job has.
    description: A message describing why the system decided to adjust the
      current number of workers, why it failed, or why the system decided to
      not make any changes to the number of workers.
    eventType: The type of autoscaling event to report.
    targetNumWorkers: The target number of workers the worker pool wants to
      resize to use.
    time: The time this event was emitted to indicate a new target or current
      num_workers value.
    workerPool: A short and friendly name for the worker pool this event
      refers to.
  """

    class EventTypeValueValuesEnum(_messages.Enum):
        """The type of autoscaling event to report.

    Values:
      TYPE_UNKNOWN: Default type for the enum. Value should never be returned.
      TARGET_NUM_WORKERS_CHANGED: The TARGET_NUM_WORKERS_CHANGED type should
        be used when the target worker pool size has changed at the start of
        an actuation. An event should always be specified as
        TARGET_NUM_WORKERS_CHANGED if it reflects a change in the
        target_num_workers.
      CURRENT_NUM_WORKERS_CHANGED: The CURRENT_NUM_WORKERS_CHANGED type should
        be used when actual worker pool size has been changed, but the
        target_num_workers has not changed.
      ACTUATION_FAILURE: The ACTUATION_FAILURE type should be used when we
        want to report an error to the user indicating why the current number
        of workers in the pool could not be changed. Displayed in the current
        status and history widgets.
      NO_CHANGE: Used when we want to report to the user a reason why we are
        not currently adjusting the number of workers. Should specify both
        target_num_workers, current_num_workers and a decision_message.
    """
        TYPE_UNKNOWN = 0
        TARGET_NUM_WORKERS_CHANGED = 1
        CURRENT_NUM_WORKERS_CHANGED = 2
        ACTUATION_FAILURE = 3
        NO_CHANGE = 4
    currentNumWorkers = _messages.IntegerField(1)
    description = _messages.MessageField('StructuredMessage', 2)
    eventType = _messages.EnumField('EventTypeValueValuesEnum', 3)
    targetNumWorkers = _messages.IntegerField(4)
    time = _messages.StringField(5)
    workerPool = _messages.StringField(6)