from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExecutorStatusValueValuesEnum(_messages.Enum):
    """Optional. Filter to select whether active/ dead or all executors
    should be selected.

    Values:
      EXECUTOR_STATUS_UNSPECIFIED: <no description>
      EXECUTOR_STATUS_ACTIVE: <no description>
      EXECUTOR_STATUS_DEAD: <no description>
    """
    EXECUTOR_STATUS_UNSPECIFIED = 0
    EXECUTOR_STATUS_ACTIVE = 1
    EXECUTOR_STATUS_DEAD = 2