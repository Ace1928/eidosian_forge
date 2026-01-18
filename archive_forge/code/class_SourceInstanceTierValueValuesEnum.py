from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourceInstanceTierValueValuesEnum(_messages.Enum):
    """Output only. The service tier of the source Filestore instance that
    this backup is created from.

    Values:
      TIER_UNSPECIFIED: Not set.
      STANDARD: STANDARD tier. BASIC_HDD is the preferred term for this tier.
      PREMIUM: PREMIUM tier. BASIC_SSD is the preferred term for this tier.
      BASIC_HDD: BASIC instances offer a maximum capacity of 63.9 TB.
        BASIC_HDD is an alias for STANDARD Tier, offering economical
        performance backed by HDD.
      BASIC_SSD: BASIC instances offer a maximum capacity of 63.9 TB.
        BASIC_SSD is an alias for PREMIUM Tier, and offers improved
        performance backed by SSD.
      HIGH_SCALE_SSD: HIGH_SCALE instances offer expanded capacity and
        performance scaling capabilities.
      ENTERPRISE: ENTERPRISE instances offer the features and availability
        needed for mission-critical workloads.
      ZONAL: ZONAL instances offer expanded capacity and performance scaling
        capabilities.
      REGIONAL: REGIONAL instances offer the features and availability needed
        for mission-critical workloads.
    """
    TIER_UNSPECIFIED = 0
    STANDARD = 1
    PREMIUM = 2
    BASIC_HDD = 3
    BASIC_SSD = 4
    HIGH_SCALE_SSD = 5
    ENTERPRISE = 6
    ZONAL = 7
    REGIONAL = 8