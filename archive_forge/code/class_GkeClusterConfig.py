from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeClusterConfig(_messages.Message):
    """The cluster's GKE config.

  Fields:
    gkeClusterTarget: Optional. A target GKE cluster to deploy to. It must be
      in the same project and region as the Dataproc cluster (the GKE cluster
      can be zonal or regional). Format:
      'projects/{project}/locations/{location}/clusters/{cluster_id}'
    namespacedGkeDeploymentTarget: Optional. Deprecated. Use gkeClusterTarget.
      Used only for the deprecated beta. A target for the deployment.
    nodePoolTarget: Optional. GKE node pools where workloads will be
      scheduled. At least one node pool must be assigned the DEFAULT
      GkeNodePoolTarget.Role. If a GkeNodePoolTarget is not specified,
      Dataproc constructs a DEFAULT GkeNodePoolTarget. Each role can be given
      to only one GkeNodePoolTarget. All node pools must have the same
      location settings.
  """
    gkeClusterTarget = _messages.StringField(1)
    namespacedGkeDeploymentTarget = _messages.MessageField('NamespacedGkeDeploymentTarget', 2)
    nodePoolTarget = _messages.MessageField('GkeNodePoolTarget', 3, repeated=True)