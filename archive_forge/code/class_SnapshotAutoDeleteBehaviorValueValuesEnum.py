from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SnapshotAutoDeleteBehaviorValueValuesEnum(_messages.Enum):
    """The behavior to use when snapshot reserved space is full.

    Values:
      SNAPSHOT_AUTO_DELETE_BEHAVIOR_UNSPECIFIED: The unspecified behavior.
      DISABLED: Don't delete any snapshots. This disables new snapshot
        creation, as long as the snapshot reserved space is full.
      OLDEST_FIRST: Delete the oldest snapshots first.
      NEWEST_FIRST: Delete the newest snapshots first.
    """
    SNAPSHOT_AUTO_DELETE_BEHAVIOR_UNSPECIFIED = 0
    DISABLED = 1
    OLDEST_FIRST = 2
    NEWEST_FIRST = 3