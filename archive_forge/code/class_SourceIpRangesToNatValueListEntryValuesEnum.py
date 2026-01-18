from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourceIpRangesToNatValueListEntryValuesEnum(_messages.Enum):
    """SourceIpRangesToNatValueListEntryValuesEnum enum type.

    Values:
      ALL_IP_RANGES: The primary and all the secondary ranges are allowed to
        Nat.
      LIST_OF_SECONDARY_IP_RANGES: A list of secondary ranges are allowed to
        Nat.
      PRIMARY_IP_RANGE: The primary range is allowed to Nat.
    """
    ALL_IP_RANGES = 0
    LIST_OF_SECONDARY_IP_RANGES = 1
    PRIMARY_IP_RANGE = 2