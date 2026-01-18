from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HierarchyLimitWarningValueValuesEnum(_messages.Enum):
    """Somewhere in hierarchy a limit is close to full. Readonly

    Values:
      HIERARCHY_LIMIT_WARNING_UNSPECIFIED: <no description>
      NO_WARNING: <no description>
      WARNING: <no description>
    """
    HIERARCHY_LIMIT_WARNING_UNSPECIFIED = 0
    NO_WARNING = 1
    WARNING = 2