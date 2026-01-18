from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ValueValueValuesEnum(_messages.Enum):
    """ValueValueValuesEnum enum type.

        Values:
          JOB_EXECUTION_STATUS_UNSPECIFIED: <no description>
          JOB_EXECUTION_STATUS_RUNNING: <no description>
          JOB_EXECUTION_STATUS_SUCCEEDED: <no description>
          JOB_EXECUTION_STATUS_FAILED: <no description>
          JOB_EXECUTION_STATUS_UNKNOWN: <no description>
        """
    JOB_EXECUTION_STATUS_UNSPECIFIED = 0
    JOB_EXECUTION_STATUS_RUNNING = 1
    JOB_EXECUTION_STATUS_SUCCEEDED = 2
    JOB_EXECUTION_STATUS_FAILED = 3
    JOB_EXECUTION_STATUS_UNKNOWN = 4