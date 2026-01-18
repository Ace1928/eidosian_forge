from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NodePool(_messages.Message):
    """NodePool contains the name and configuration for a cluster's node pool.
  Node pools are a set of nodes (i.e. VM's), with a common configuration and
  specification, under the control of the cluster master. They may have a set
  of Kubernetes labels applied to them, which may be used to reference them
  during pod scheduling. They may also be resized up or down, to accommodate
  the workload.

  Enums:
    StatusValueValuesEnum: [Output only] The status of the nodes in this pool
      instance.

  Fields:
    autoscaling: Autoscaler configuration for this NodePool. Autoscaler is
      enabled only if a valid configuration is present.
    bestEffortProvisioning: Enable best effort provisioning for nodes
    conditions: Which conditions caused the current node pool state.
    config: The node configuration of the pool.
    etag: This checksum is computed by the server based on the value of node
      pool fields, and may be sent on update requests to ensure the client has
      an up-to-date value before proceeding.
    initialNodeCount: The initial node count for the pool. You must ensure
      that your Compute Engine [resource
      quota](https://cloud.google.com/compute/quotas) is sufficient for this
      number of instances. You must also have available firewall and routes
      quota.
    instanceGroupUrls: [Output only] The resource URLs of the [managed
      instance groups](https://cloud.google.com/compute/docs/instance-
      groups/creating-groups-of-managed-instances) associated with this node
      pool. During the node pool blue-green upgrade operation, the URLs
      contain both blue and green resources.
    locations: The list of Google Compute Engine
      [zones](https://cloud.google.com/compute/docs/zones#available) in which
      the NodePool's nodes should be located. If this value is unspecified
      during node pool creation, the
      [Cluster.Locations](https://cloud.google.com/kubernetes-engine/docs/refe
      rence/rest/v1/projects.locations.clusters#Cluster.FIELDS.locations)
      value will be used, instead. Warning: changing node pool locations will
      result in nodes being added and/or removed.
    management: NodeManagement configuration for this NodePool.
    maxPodsConstraint: The constraint on the maximum number of pods that can
      be run simultaneously on a node in the node pool.
    name: The name of the node pool.
    networkConfig: Networking configuration for this NodePool. If specified,
      it overrides the cluster-level defaults.
    placementPolicy: Specifies the node placement policy.
    podIpv4CidrSize: [Output only] The pod CIDR block size per node in this
      node pool.
    queuedProvisioning: Specifies the configuration of queued provisioning.
    selfLink: [Output only] Server-defined URL for the resource.
    status: [Output only] The status of the nodes in this pool instance.
    statusMessage: [Output only] Deprecated. Use conditions instead.
      Additional information about the current status of this node pool
      instance, if available.
    updateInfo: Output only. [Output only] Update info contains relevant
      information during a node pool update.
    upgradeSettings: Upgrade settings control disruption and speed of the
      upgrade.
    version: The version of Kubernetes running on this NodePool's nodes. If
      unspecified, it defaults as described
      [here](https://cloud.google.com/kubernetes-
      engine/versioning#specifying_node_version).
  """

    class StatusValueValuesEnum(_messages.Enum):
        """[Output only] The status of the nodes in this pool instance.

    Values:
      STATUS_UNSPECIFIED: Not set.
      PROVISIONING: The PROVISIONING state indicates the node pool is being
        created.
      RUNNING: The RUNNING state indicates the node pool has been created and
        is fully usable.
      RUNNING_WITH_ERROR: The RUNNING_WITH_ERROR state indicates the node pool
        has been created and is partially usable. Some error state has
        occurred and some functionality may be impaired. Customer may need to
        reissue a request or trigger a new update.
      RECONCILING: The RECONCILING state indicates that some work is actively
        being done on the node pool, such as upgrading node software. Details
        can be found in the `statusMessage` field.
      STOPPING: The STOPPING state indicates the node pool is being deleted.
      ERROR: The ERROR state indicates the node pool may be unusable. Details
        can be found in the `statusMessage` field.
    """
        STATUS_UNSPECIFIED = 0
        PROVISIONING = 1
        RUNNING = 2
        RUNNING_WITH_ERROR = 3
        RECONCILING = 4
        STOPPING = 5
        ERROR = 6
    autoscaling = _messages.MessageField('NodePoolAutoscaling', 1)
    bestEffortProvisioning = _messages.MessageField('BestEffortProvisioning', 2)
    conditions = _messages.MessageField('StatusCondition', 3, repeated=True)
    config = _messages.MessageField('NodeConfig', 4)
    etag = _messages.StringField(5)
    initialNodeCount = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    instanceGroupUrls = _messages.StringField(7, repeated=True)
    locations = _messages.StringField(8, repeated=True)
    management = _messages.MessageField('NodeManagement', 9)
    maxPodsConstraint = _messages.MessageField('MaxPodsConstraint', 10)
    name = _messages.StringField(11)
    networkConfig = _messages.MessageField('NodeNetworkConfig', 12)
    placementPolicy = _messages.MessageField('PlacementPolicy', 13)
    podIpv4CidrSize = _messages.IntegerField(14, variant=_messages.Variant.INT32)
    queuedProvisioning = _messages.MessageField('QueuedProvisioning', 15)
    selfLink = _messages.StringField(16)
    status = _messages.EnumField('StatusValueValuesEnum', 17)
    statusMessage = _messages.StringField(18)
    updateInfo = _messages.MessageField('UpdateInfo', 19)
    upgradeSettings = _messages.MessageField('UpgradeSettings', 20)
    version = _messages.StringField(21)