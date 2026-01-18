from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NamespacedGkeDeploymentTarget(_messages.Message):
    """Deprecated. Used only for the deprecated beta. A full, namespace-
  isolated deployment target for an existing GKE cluster.

  Fields:
    clusterNamespace: Optional. A namespace within the GKE cluster to deploy
      into.
    targetGkeCluster: Optional. The target GKE cluster to deploy to. Format:
      'projects/{project}/locations/{location}/clusters/{cluster_id}'
  """
    clusterNamespace = _messages.StringField(1)
    targetGkeCluster = _messages.StringField(2)