from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TotalEgressBandwidthTierValueValuesEnum(_messages.Enum):
    """Optional. The tier of the total egress bandwidth.

    Values:
      TIER_UNSPECIFIED: This value is unused.
      DEFAULT: Default network performance config.
      TIER_1: Tier 1 network performance config.
    """
    TIER_UNSPECIFIED = 0
    DEFAULT = 1
    TIER_1 = 2