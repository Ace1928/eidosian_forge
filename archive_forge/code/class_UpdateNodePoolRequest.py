from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateNodePoolRequest(_messages.Message):
    """UpdateNodePoolRequests update a node pool's image and/or version.

  Fields:
    accelerators: A list of hardware accelerators to be attached to each node.
      See https://cloud.google.com/compute/docs/gpus for more information
      about support for GPUs.
    clusterId: Deprecated. The name of the cluster to upgrade. This field has
      been deprecated and replaced by the name field.
    confidentialNodes: Confidential nodes config. All the nodes in the node
      pool will be Confidential VM once enabled.
    containerdConfig: The desired containerd config for nodes in the node
      pool. Initiates an upgrade operation that recreates the nodes with the
      new config.
    diskSizeGb: Optional. The desired disk size for nodes in the node pool
      specified in GB. The smallest allowed disk size is 10GB. Initiates an
      upgrade operation that migrates the nodes in the node pool to the
      specified disk size.
    diskType: Optional. The desired disk type (e.g. 'pd-standard', 'pd-ssd' or
      'pd-balanced') for nodes in the node pool. Initiates an upgrade
      operation that migrates the nodes in the node pool to the specified disk
      type.
    etag: The current etag of the node pool. If an etag is provided and does
      not match the current etag of the node pool, update will be blocked and
      an ABORTED error will be returned.
    fastSocket: Enable or disable NCCL fast socket for the node pool.
    gcfsConfig: GCFS config.
    gvnic: Enable or disable gvnic on the node pool.
    image: The desired name of the image name to use for this node. This is
      used to create clusters using a custom image.
    imageProject: The project containing the desired image to use for this
      node pool. This is used to create clusters using a custom image.
    imageType: Required. The desired image type for the node pool. Please see
      https://cloud.google.com/kubernetes-engine/docs/concepts/node-images for
      available image types.
    kubeletConfig: Node kubelet configs.
    labels: The desired node labels to be applied to all nodes in the node
      pool. If this field is not present, the labels will not be changed.
      Otherwise, the existing node labels will be *replaced* with the provided
      labels.
    linuxNodeConfig: Parameters that can be configured on Linux nodes.
    locations: The desired list of Google Compute Engine
      [zones](https://cloud.google.com/compute/docs/zones#available) in which
      the node pool's nodes should be located. Changing the locations for a
      node pool will result in nodes being either created or removed from the
      node pool, depending on whether locations are being added or removed.
    loggingConfig: Logging configuration.
    machineType: Optional. The desired [Google Compute Engine machine
      type](https://cloud.google.com/compute/docs/machine-types) for nodes in
      the node pool. Initiates an upgrade operation that migrates the nodes in
      the node pool to the specified machine type.
    name: The name (project, location, cluster, node pool) of the node pool to
      update. Specified in the format
      `projects/*/locations/*/clusters/*/nodePools/*`.
    nodeNetworkConfig: Node network config.
    nodePoolId: Deprecated. The name of the node pool to upgrade. This field
      has been deprecated and replaced by the name field.
    nodeVersion: Required. The Kubernetes version to change the nodes to
      (typically an upgrade). Users may specify either explicit versions
      offered by Kubernetes Engine or version aliases, which have the
      following behavior: - "latest": picks the highest valid Kubernetes
      version - "1.X": picks the highest valid patch+gke.N patch in the 1.X
      version - "1.X.Y": picks the highest valid gke.N patch in the 1.X.Y
      version - "1.X.Y-gke.N": picks an explicit Kubernetes version - "-":
      picks the Kubernetes master version
    projectId: Deprecated. The Google Developers Console [project ID or
      project number](https://cloud.google.com/resource-manager/docs/creating-
      managing-projects). This field has been deprecated and replaced by the
      name field.
    queuedProvisioning: Specifies the configuration of queued provisioning.
    resourceLabels: The resource labels for the node pool to use to annotate
      any related Google Compute Engine resources.
    resourceManagerTags: Desired resource manager tag keys and values to be
      attached to the nodes for managing Compute Engine firewalls using
      Network Firewall Policies. Existing tags will be replaced with new
      values.
    storagePools: List of Storage Pools where boot disks are provisioned.
      Existing Storage Pools will be replaced with storage-pools.
    tags: The desired network tags to be applied to all nodes in the node
      pool. If this field is not present, the tags will not be changed.
      Otherwise, the existing network tags will be *replaced* with the
      provided tags.
    taints: The desired node taints to be applied to all nodes in the node
      pool. If this field is not present, the taints will not be changed.
      Otherwise, the existing node taints will be *replaced* with the provided
      taints.
    upgradeSettings: Upgrade settings control disruption and speed of the
      upgrade.
    windowsNodeConfig: Parameters that can be configured on Windows nodes.
    workloadMetadataConfig: The desired workload metadata config for the node
      pool.
    zone: Deprecated. The name of the Google Compute Engine
      [zone](https://cloud.google.com/compute/docs/zones#available) in which
      the cluster resides. This field has been deprecated and replaced by the
      name field.
  """
    accelerators = _messages.MessageField('AcceleratorConfig', 1, repeated=True)
    clusterId = _messages.StringField(2)
    confidentialNodes = _messages.MessageField('ConfidentialNodes', 3)
    containerdConfig = _messages.MessageField('ContainerdConfig', 4)
    diskSizeGb = _messages.IntegerField(5)
    diskType = _messages.StringField(6)
    etag = _messages.StringField(7)
    fastSocket = _messages.MessageField('FastSocket', 8)
    gcfsConfig = _messages.MessageField('GcfsConfig', 9)
    gvnic = _messages.MessageField('VirtualNIC', 10)
    image = _messages.StringField(11)
    imageProject = _messages.StringField(12)
    imageType = _messages.StringField(13)
    kubeletConfig = _messages.MessageField('NodeKubeletConfig', 14)
    labels = _messages.MessageField('NodeLabels', 15)
    linuxNodeConfig = _messages.MessageField('LinuxNodeConfig', 16)
    locations = _messages.StringField(17, repeated=True)
    loggingConfig = _messages.MessageField('NodePoolLoggingConfig', 18)
    machineType = _messages.StringField(19)
    name = _messages.StringField(20)
    nodeNetworkConfig = _messages.MessageField('NodeNetworkConfig', 21)
    nodePoolId = _messages.StringField(22)
    nodeVersion = _messages.StringField(23)
    projectId = _messages.StringField(24)
    queuedProvisioning = _messages.MessageField('QueuedProvisioning', 25)
    resourceLabels = _messages.MessageField('ResourceLabels', 26)
    resourceManagerTags = _messages.MessageField('ResourceManagerTags', 27)
    storagePools = _messages.StringField(28, repeated=True)
    tags = _messages.MessageField('NetworkTags', 29)
    taints = _messages.MessageField('NodeTaints', 30)
    upgradeSettings = _messages.MessageField('UpgradeSettings', 31)
    windowsNodeConfig = _messages.MessageField('WindowsNodeConfig', 32)
    workloadMetadataConfig = _messages.MessageField('WorkloadMetadataConfig', 33)
    zone = _messages.StringField(34)