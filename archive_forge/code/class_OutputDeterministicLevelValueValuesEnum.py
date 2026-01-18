from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OutputDeterministicLevelValueValuesEnum(_messages.Enum):
    """OutputDeterministicLevelValueValuesEnum enum type.

    Values:
      DETERMINISTIC_LEVEL_UNSPECIFIED: <no description>
      DETERMINISTIC_LEVEL_DETERMINATE: <no description>
      DETERMINISTIC_LEVEL_UNORDERED: <no description>
      DETERMINISTIC_LEVEL_INDETERMINATE: <no description>
    """
    DETERMINISTIC_LEVEL_UNSPECIFIED = 0
    DETERMINISTIC_LEVEL_DETERMINATE = 1
    DETERMINISTIC_LEVEL_UNORDERED = 2
    DETERMINISTIC_LEVEL_INDETERMINATE = 3