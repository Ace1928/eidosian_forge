from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterTierValueValuesEnum(_messages.Enum):
    """Output only. [Output only] cluster_tier specifies the premium tier of
    the cluster.

    Values:
      CLUSTER_TIER_UNSPECIFIED: CLUSTER_TIER_UNSPECIFIED is when cluster_tier
        is not set.
      STANDARD: STANDARD indicates a standard GKE cluster.
      ENTERPRISE: ENTERPRISE indicates a GKE Enterprise cluster.
    """
    CLUSTER_TIER_UNSPECIFIED = 0
    STANDARD = 1
    ENTERPRISE = 2