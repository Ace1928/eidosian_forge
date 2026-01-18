from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContainsPypiModulesConflictValueValuesEnum(_messages.Enum):
    """Output only. Whether build has succeeded or failed on modules
    conflicts.

    Values:
      CONFLICT_RESULT_UNSPECIFIED: It is unknown whether build had conflicts
        or not.
      CONFLICT: There were python packages conflicts.
      NO_CONFLICT: There were no python packages conflicts.
    """
    CONFLICT_RESULT_UNSPECIFIED = 0
    CONFLICT = 1
    NO_CONFLICT = 2