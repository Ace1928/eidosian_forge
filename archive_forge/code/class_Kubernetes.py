from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Kubernetes(_messages.Message):
    """Kubernetes-related attributes.

  Fields:
    accessReviews: Provides information on any Kubernetes access reviews
      (privilege checks) relevant to the finding.
    bindings: Provides Kubernetes role binding information for findings that
      involve [RoleBindings or
      ClusterRoleBindings](https://cloud.google.com/kubernetes-
      engine/docs/how-to/role-based-access-control).
    nodePools: GKE [node pools](https://cloud.google.com/kubernetes-
      engine/docs/concepts/node-pools) associated with the finding. This field
      contains node pool information for each node, when it is available.
    nodes: Provides Kubernetes [node](https://cloud.google.com/kubernetes-
      engine/docs/concepts/cluster-architecture#nodes) information.
    objects: Kubernetes objects related to the finding.
    pods: Kubernetes [Pods](https://cloud.google.com/kubernetes-
      engine/docs/concepts/pod) associated with the finding. This field
      contains Pod records for each container that is owned by a Pod.
    roles: Provides Kubernetes role information for findings that involve
      [Roles or ClusterRoles](https://cloud.google.com/kubernetes-
      engine/docs/how-to/role-based-access-control).
  """
    accessReviews = _messages.MessageField('AccessReview', 1, repeated=True)
    bindings = _messages.MessageField('GoogleCloudSecuritycenterV1Binding', 2, repeated=True)
    nodePools = _messages.MessageField('NodePool', 3, repeated=True)
    nodes = _messages.MessageField('Node', 4, repeated=True)
    objects = _messages.MessageField('Object', 5, repeated=True)
    pods = _messages.MessageField('Pod', 6, repeated=True)
    roles = _messages.MessageField('Role', 7, repeated=True)