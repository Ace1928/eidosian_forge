from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkItemDetails(_messages.Message):
    """Information about an individual work item execution.

  Enums:
    StateValueValuesEnum: State of this work item.

  Fields:
    attemptId: Attempt ID of this work item
    endTime: End time of this work item attempt. If the work item is
      completed, this is the actual end time of the work item. Otherwise, it
      is the predicted end time.
    metrics: Metrics for this work item.
    progress: Progress of this work item.
    startTime: Start time of this work item attempt.
    state: State of this work item.
    stragglerInfo: Information about straggler detections for this work item.
    taskId: Name of this work item.
  """

    class StateValueValuesEnum(_messages.Enum):
        """State of this work item.

    Values:
      EXECUTION_STATE_UNKNOWN: The component state is unknown or unspecified.
      EXECUTION_STATE_NOT_STARTED: The component is not yet running.
      EXECUTION_STATE_RUNNING: The component is currently running.
      EXECUTION_STATE_SUCCEEDED: The component succeeded.
      EXECUTION_STATE_FAILED: The component failed.
      EXECUTION_STATE_CANCELLED: Execution of the component was cancelled.
    """
        EXECUTION_STATE_UNKNOWN = 0
        EXECUTION_STATE_NOT_STARTED = 1
        EXECUTION_STATE_RUNNING = 2
        EXECUTION_STATE_SUCCEEDED = 3
        EXECUTION_STATE_FAILED = 4
        EXECUTION_STATE_CANCELLED = 5
    attemptId = _messages.StringField(1)
    endTime = _messages.StringField(2)
    metrics = _messages.MessageField('MetricUpdate', 3, repeated=True)
    progress = _messages.MessageField('ProgressTimeseries', 4)
    startTime = _messages.StringField(5)
    state = _messages.EnumField('StateValueValuesEnum', 6)
    stragglerInfo = _messages.MessageField('StragglerInfo', 7)
    taskId = _messages.StringField(8)