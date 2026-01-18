from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TierValueValuesEnum(_messages.Enum):
    """Output only. The CaPool.Tier of the CaPool that includes this
    CertificateAuthority.

    Values:
      TIER_UNSPECIFIED: Not specified.
      ENTERPRISE: Enterprise tier.
      DEVOPS: DevOps tier.
    """
    TIER_UNSPECIFIED = 0
    ENTERPRISE = 1
    DEVOPS = 2