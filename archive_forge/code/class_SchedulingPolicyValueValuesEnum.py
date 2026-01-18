from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SchedulingPolicyValueValuesEnum(_messages.Enum):
    """Scheduling policy for Tasks in the TaskGroup. The default value is
    AS_SOON_AS_POSSIBLE.

    Values:
      SCHEDULING_POLICY_UNSPECIFIED: Unspecified.
      AS_SOON_AS_POSSIBLE: Run Tasks as soon as resources are available. Tasks
        might be executed in parallel depending on parallelism and task_count
        values.
      IN_ORDER: Run Tasks sequentially with increased task index.
    """
    SCHEDULING_POLICY_UNSPECIFIED = 0
    AS_SOON_AS_POSSIBLE = 1
    IN_ORDER = 2