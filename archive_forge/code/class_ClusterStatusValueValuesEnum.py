from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterStatusValueValuesEnum(_messages.Enum):
    """The status of cluster underlying the membership.

    Values:
      CLUSTER_STATUS_UNSPECIFIED: The cluster status is unspecified.
      CLUSTER_ACTIVE: The cluster is active.
      CLUSTER_INACTIVE: The cluster is inactive.
    """
    CLUSTER_STATUS_UNSPECIFIED = 0
    CLUSTER_ACTIVE = 1
    CLUSTER_INACTIVE = 2