from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExpectedStructureValueValuesEnum(_messages.Enum):
    """The issue type of InvalidDataPartition.

    Values:
      PARTITION_STRUCTURE_UNSPECIFIED: PartitionStructure unspecified.
      CONSISTENT_KEYS: Consistent hive-style partition definition (both raw
        and curated zone).
      HIVE_STYLE_KEYS: Hive style partition definition (curated zone only).
    """
    PARTITION_STRUCTURE_UNSPECIFIED = 0
    CONSISTENT_KEYS = 1
    HIVE_STYLE_KEYS = 2