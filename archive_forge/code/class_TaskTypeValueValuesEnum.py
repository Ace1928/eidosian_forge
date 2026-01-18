from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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