from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetStatusValueValuesEnum(_messages.Enum):
    """[Output Only] The eventual status of the instance. The instance group
    manager will not be identified as stable till each managed instance
    reaches its targetStatus.

    Values:
      ABANDONED: The managed instance will eventually be ABANDONED, i.e.
        dissociated from the managed instance group.
      DELETED: The managed instance will eventually be DELETED.
      RUNNING: The managed instance will eventually reach status RUNNING.
      STOPPED: The managed instance will eventually reach status TERMINATED.
      SUSPENDED: The managed instance will eventually reach status SUSPENDED.
    """
    ABANDONED = 0
    DELETED = 1
    RUNNING = 2
    STOPPED = 3
    SUSPENDED = 4