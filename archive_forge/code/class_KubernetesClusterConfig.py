from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KubernetesClusterConfig(_messages.Message):
    """The configuration for running the Dataproc cluster on Kubernetes.

  Fields:
    gdceClusterConfig: Required. The configuration for running the Dataproc
      cluster on GDCE.
    gkeClusterConfig: Required. The configuration for running the Dataproc
      cluster on GKE.
    kubernetesNamespace: Optional. A namespace within the Kubernetes cluster
      to deploy into. If this namespace does not exist, it is created. If it
      exists, Dataproc verifies that another Dataproc VirtualCluster is not
      installed into it. If not specified, the name of the Dataproc Cluster is
      used.
    kubernetesSoftwareConfig: Optional. The software configuration for this
      Dataproc cluster running on Kubernetes.
  """
    gdceClusterConfig = _messages.MessageField('GdceClusterConfig', 1)
    gkeClusterConfig = _messages.MessageField('GkeClusterConfig', 2)
    kubernetesNamespace = _messages.StringField(3)
    kubernetesSoftwareConfig = _messages.MessageField('KubernetesSoftwareConfig', 4)