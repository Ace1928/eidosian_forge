from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OnPremCluster(_messages.Message):
    """OnPremCluster contains information specific to GKE On-Prem clusters.

  Enums:
    ClusterTypeValueValuesEnum: Immutable. The on prem cluster's type.

  Fields:
    adminCluster: Immutable. Whether the cluster is an admin cluster.
    clusterMissing: Output only. If cluster_missing is set then it denotes
      that API(gkeonprem.googleapis.com) resource for this GKE On-Prem cluster
      no longer exists.
    clusterType: Immutable. The on prem cluster's type.
    resourceLink: Immutable. Self-link of the Google Cloud resource for the
      GKE On-Prem cluster. For example:
      //gkeonprem.googleapis.com/projects/my-project/locations/us-
      west1-a/vmwareClusters/my-cluster
      //gkeonprem.googleapis.com/projects/my-project/locations/us-
      west1-a/bareMetalClusters/my-cluster
  """

    class ClusterTypeValueValuesEnum(_messages.Enum):
        """Immutable. The on prem cluster's type.

    Values:
      CLUSTERTYPE_UNSPECIFIED: The ClusterType is not set.
      BOOTSTRAP: The ClusterType is bootstrap cluster.
      HYBRID: The ClusterType is baremetal hybrid cluster.
      STANDALONE: The ClusterType is baremetal standalone cluster.
      USER: The ClusterType is user cluster.
    """
        CLUSTERTYPE_UNSPECIFIED = 0
        BOOTSTRAP = 1
        HYBRID = 2
        STANDALONE = 3
        USER = 4
    adminCluster = _messages.BooleanField(1)
    clusterMissing = _messages.BooleanField(2)
    clusterType = _messages.EnumField('ClusterTypeValueValuesEnum', 3)
    resourceLink = _messages.StringField(4)