from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StreamingComputationTask(_messages.Message):
    """A task which describes what action should be performed for the specified
  streaming computation ranges.

  Enums:
    TaskTypeValueValuesEnum: A type of streaming computation task.

  Fields:
    computationRanges: Contains ranges of a streaming computation this task
      should apply to.
    dataDisks: Describes the set of data disks this task should apply to.
    taskType: A type of streaming computation task.
  """

    class TaskTypeValueValuesEnum(_messages.Enum):
        """A type of streaming computation task.

    Values:
      STREAMING_COMPUTATION_TASK_UNKNOWN: The streaming computation task is
        unknown, or unspecified.
      STREAMING_COMPUTATION_TASK_STOP: Stop processing specified streaming
        computation range(s).
      STREAMING_COMPUTATION_TASK_START: Start processing specified streaming
        computation range(s).
    """
        STREAMING_COMPUTATION_TASK_UNKNOWN = 0
        STREAMING_COMPUTATION_TASK_STOP = 1
        STREAMING_COMPUTATION_TASK_START = 2
    computationRanges = _messages.MessageField('StreamingComputationRanges', 1, repeated=True)
    dataDisks = _messages.MessageField('MountedDataDisk', 2, repeated=True)
    taskType = _messages.EnumField('TaskTypeValueValuesEnum', 3)