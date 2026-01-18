from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AddonsConfig(_messages.Message):
    """Configuration for the addons that can be automatically spun up in the
  cluster, enabling additional functionality.

  Fields:
    cloudRunConfig: Configuration for the Cloud Run addon, which allows the
      user to use a managed Knative service.
    configConnectorConfig: Configuration for the ConfigConnector add-on, a
      Kubernetes extension to manage hosted GCP services through the
      Kubernetes API
    dnsCacheConfig: Configuration for NodeLocalDNS, a dns cache running on
      cluster nodes
    gcePersistentDiskCsiDriverConfig: Configuration for the Compute Engine
      Persistent Disk CSI driver.
    gcpFilestoreCsiDriverConfig: Configuration for the GCP Filestore CSI
      driver.
    gcsFuseCsiDriverConfig: Configuration for the Cloud Storage Fuse CSI
      driver.
    gkeBackupAgentConfig: Configuration for the Backup for GKE agent addon.
    horizontalPodAutoscaling: Configuration for the horizontal pod autoscaling
      feature, which increases or decreases the number of replica pods a
      replication controller has based on the resource usage of the existing
      pods.
    httpLoadBalancing: Configuration for the HTTP (L7) load balancing
      controller addon, which makes it easy to set up HTTP load balancers for
      services in a cluster.
    kubernetesDashboard: Configuration for the Kubernetes Dashboard. This
      addon is deprecated, and will be disabled in 1.15. It is recommended to
      use the Cloud Console to manage and monitor your Kubernetes clusters,
      workloads and applications. For more information, see:
      https://cloud.google.com/kubernetes-engine/docs/concepts/dashboards
    networkPolicyConfig: Configuration for NetworkPolicy. This only tracks
      whether the addon is enabled or not on the Master, it does not track
      whether network policy is enabled for the nodes.
    parallelstoreCsiDriverConfig: Configuration for the Cloud Storage
      Parallelstore CSI driver.
    rayConfig: Optional. Configuration for Ray addon.
    statefulHaConfig: Optional. Configuration for the StatefulHA add-on.
  """
    cloudRunConfig = _messages.MessageField('CloudRunConfig', 1)
    configConnectorConfig = _messages.MessageField('ConfigConnectorConfig', 2)
    dnsCacheConfig = _messages.MessageField('DnsCacheConfig', 3)
    gcePersistentDiskCsiDriverConfig = _messages.MessageField('GcePersistentDiskCsiDriverConfig', 4)
    gcpFilestoreCsiDriverConfig = _messages.MessageField('GcpFilestoreCsiDriverConfig', 5)
    gcsFuseCsiDriverConfig = _messages.MessageField('GcsFuseCsiDriverConfig', 6)
    gkeBackupAgentConfig = _messages.MessageField('GkeBackupAgentConfig', 7)
    horizontalPodAutoscaling = _messages.MessageField('HorizontalPodAutoscaling', 8)
    httpLoadBalancing = _messages.MessageField('HttpLoadBalancing', 9)
    kubernetesDashboard = _messages.MessageField('KubernetesDashboard', 10)
    networkPolicyConfig = _messages.MessageField('NetworkPolicyConfig', 11)
    parallelstoreCsiDriverConfig = _messages.MessageField('ParallelstoreCsiDriverConfig', 12)
    rayConfig = _messages.MessageField('RayConfig', 13)
    statefulHaConfig = _messages.MessageField('StatefulHAConfig', 14)