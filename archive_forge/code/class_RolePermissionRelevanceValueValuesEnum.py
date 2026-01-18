from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RolePermissionRelevanceValueValuesEnum(_messages.Enum):
    """The relevance of the permission's existence, or nonexistence, in the
    role to the overall determination for the entire policy.

    Values:
      HEURISTIC_RELEVANCE_UNSPECIFIED: Default value. This value is unused.
      NORMAL: The data point has a limited effect on the result. Changing the
        data point is unlikely to affect the overall determination.
      HIGH: The data point has a strong effect on the result. Changing the
        data point is likely to affect the overall determination.
    """
    HEURISTIC_RELEVANCE_UNSPECIFIED = 0
    NORMAL = 1
    HIGH = 2