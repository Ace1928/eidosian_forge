from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StageSummary(_messages.Message):
    """Information about a particular execution stage of a job.

  Enums:
    StateValueValuesEnum: State of this stage.

  Fields:
    endTime: End time of this stage. If the work item is completed, this is
      the actual end time of the stage. Otherwise, it is the predicted end
      time.
    metrics: Metrics for this stage.
    progress: Progress for this stage. Only applicable to Batch jobs.
    stageId: ID of this stage
    startTime: Start time of this stage.
    state: State of this stage.
    stragglerSummary: Straggler summary for this stage.
  """

    class StateValueValuesEnum(_messages.Enum):
        """State of this stage.

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
    endTime = _messages.StringField(1)
    metrics = _messages.MessageField('MetricUpdate', 2, repeated=True)
    progress = _messages.MessageField('ProgressTimeseries', 3)
    stageId = _messages.StringField(4)
    startTime = _messages.StringField(5)
    state = _messages.EnumField('StateValueValuesEnum', 6)
    stragglerSummary = _messages.MessageField('StragglerSummary', 7)