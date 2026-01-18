from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MembershipEndpoint(_messages.Message):
    """MembershipEndpoint contains information needed to contact a Kubernetes
  API, endpoint and any additional Kubernetes metadata.

  Fields:
    applianceCluster: Optional. Specific information for a GDC Edge Appliance
      cluster.
    edgeCluster: Optional. Specific information for a Google Edge cluster.
    gkeCluster: Optional. Specific information for a GKE-on-GCP cluster.
    googleManaged: Output only. Whether the lifecycle of this membership is
      managed by a google cluster platform service.
    kubernetesMetadata: Output only. Useful Kubernetes-specific metadata.
    kubernetesResource: Optional. The in-cluster Kubernetes Resources that
      should be applied for a correctly registered cluster, in the steady
      state. These resources: * Ensure that the cluster is exclusively
      registered to one and only one Hub Membership. * Propagate Workload Pool
      Information available in the Membership Authority field. * Ensure proper
      initial configuration of default Hub Features.
    multiCloudCluster: Optional. Specific information for a GKE Multi-Cloud
      cluster.
    onPremCluster: Optional. Specific information for a GKE On-Prem cluster.
      An onprem user-cluster who has no resourceLink is not allowed to use
      this field, it should have a nil "type" instead.
  """
    applianceCluster = _messages.MessageField('ApplianceCluster', 1)
    edgeCluster = _messages.MessageField('EdgeCluster', 2)
    gkeCluster = _messages.MessageField('GkeCluster', 3)
    googleManaged = _messages.BooleanField(4)
    kubernetesMetadata = _messages.MessageField('KubernetesMetadata', 5)
    kubernetesResource = _messages.MessageField('KubernetesResource', 6)
    multiCloudCluster = _messages.MessageField('MultiCloudCluster', 7)
    onPremCluster = _messages.MessageField('OnPremCluster', 8)