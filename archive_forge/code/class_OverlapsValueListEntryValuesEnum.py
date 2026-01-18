from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OverlapsValueListEntryValuesEnum(_messages.Enum):
    """OverlapsValueListEntryValuesEnum enum type.

    Values:
      OVERLAP_UNSPECIFIED: No overlap overrides.
      OVERLAP_ROUTE_RANGE: Allow creation of static routes more specific that
        the current internal range.
      OVERLAP_EXISTING_SUBNET_RANGE: Allow creation of internal ranges that
        overlap with existing subnets.
    """
    OVERLAP_UNSPECIFIED = 0
    OVERLAP_ROUTE_RANGE = 1
    OVERLAP_EXISTING_SUBNET_RANGE = 2