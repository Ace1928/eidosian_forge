from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DumpParallelLevelValueValuesEnum(_messages.Enum):
    """Initial dump parallelism level.

    Values:
      DUMP_PARALLEL_LEVEL_UNSPECIFIED: Unknown dump parallel level. Will be
        defaulted to OPTIMAL.
      MIN: Minimal parallel level.
      OPTIMAL: Optimal parallel level.
      MAX: Maximum parallel level.
    """
    DUMP_PARALLEL_LEVEL_UNSPECIFIED = 0
    MIN = 1
    OPTIMAL = 2
    MAX = 3